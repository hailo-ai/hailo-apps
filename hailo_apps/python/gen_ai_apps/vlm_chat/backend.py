import time
import multiprocessing as mp
import numpy as np
import cv2
from hailo_platform import VDevice
from hailo_platform.genai import VLM
from hailo_apps.python.core.common.defines import SHARED_VDEVICE_GROUP_ID
from hailo_apps.python.core.common.core import get_logger

# Initialize logger
logger = get_logger(__name__)

def vlm_worker_process(request_queue: mp.Queue, response_queue: mp.Queue, hef_path: str,
                      max_tokens: int, temperature: float, seed: int) -> None:
    """
    Worker process for VLM inference.

    Args:
        request_queue (mp.Queue): Queue to receive inference requests.
        response_queue (mp.Queue): Queue to send inference results.
        hef_path (str): Path to the HEF model file.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        seed (int): Random seed.
    """
    try:
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        vlm = VLM(vdevice, hef_path)

        while True:
            item = request_queue.get()
            if item is None:
                break

            try:
                result = _hailo_inference_inner(
                    item['numpy_image'],
                    item['prompts'],
                    vlm,
                    max_tokens,
                    temperature,
                    seed
                )
                response_queue.put({'result': result, 'error': None})
            except Exception as e:
                logger.error(f"Inference error: {e}")
                response_queue.put({'result': None, 'error': str(e)})

    except Exception as e:
        logger.error(f"Worker process error: {e}")
        response_queue.put({'result': None, 'error': f"{str(e)}"})
    finally:
        try:
            vlm.release()
            vdevice.release()
        except Exception as e:
            logger.debug(f"Error releasing resources: {e}")

def _hailo_inference_inner(image: np.ndarray, prompts: dict, vlm: VLM,
                          max_tokens: int, temperature: float, seed: int) -> dict:
    """
    Inner inference function executed inside the worker process.

    Args:
        image (np.ndarray): Input image.
        prompts (dict): Dictionary containing system and user prompts.
        vlm (VLM): Initialized VLM instance.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        seed (int): Random seed.

    Returns:
        dict: Dictionary containing the answer and inference time.
    """
    try:
        response = ''
        start_time = time.time()
        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompts['system_prompt']}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompts['user_prompt']}
                ]
            }
        ]

        with vlm.generate(prompt=prompt, frames=[image], temperature=temperature, seed=seed, max_generated_tokens=max_tokens) as generation:
            for chunk in generation:
                if chunk != '<|im_end|>':
                    print(chunk, end='', flush=True)  # Keep print for real-time feedback in console
                    response += chunk

        vlm.clear_context()
        end_time = time.time()
        return {
            'answer': response.replace('<|im_end|>', '').strip(),
            'time': f"{end_time - start_time:.2f} seconds"
        }
    except Exception as e:
        logger.error(f"Inference inner error: {e}")
        return {
            'answer': f'Error: {str(e)}',
            'time': f"{time.time() - start_time:.2f} seconds"
        }

class Backend:
    """
    Backend class to handle VLM inference using multiprocessing.
    """

    def __init__(self, hef_path: str, max_tokens: int = 200, temperature: float = 0.1,
                 seed: int = 42, system_prompt: str = 'You are a helpful assistant that analyzes images and answers questions about them.') -> None:
        """
        Initialize the Backend.

        Args:
            hef_path (str): Path to the HEF model file.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 200.
            temperature (float, optional): Sampling temperature. Defaults to 0.1.
            seed (int, optional): Random seed. Defaults to 42.
            system_prompt (str, optional): System prompt for VLM. Defaults to helpful assistant prompt.
        """
        self.hef_path = hef_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.system_prompt = system_prompt

        self._request_queue = mp.Queue(maxsize=10)  # Limit queue size
        self._response_queue = mp.Queue(maxsize=10)
        self._process = mp.Process(
            target=vlm_worker_process,
            args=(self._request_queue, self._response_queue, self.hef_path, self.max_tokens, self.temperature, self.seed)
        )
        self._process.start()
        logger.info("VLM backend process started.")

    def vlm_inference(self, image: np.ndarray, prompt: str, timeout: int = 30) -> dict:
        """
        Run VLM inference on an image with a prompt.

        Args:
            image (np.ndarray): Input image.
            prompt (str): User prompt/question.
            timeout (int, optional): Timeout in seconds. Defaults to 30.

        Returns:
            dict: Inference result containing answer and time.
        """
        request_data = {
            'numpy_image': self.convert_resize_image(image),
            'prompts': {
                'system_prompt': self.system_prompt,
                'user_prompt': prompt,
            }
        }
        return self._execute_inference(request_data, timeout)

    def _execute_inference(self, request_data: dict, timeout: int) -> dict:
        """
        Execute inference request via queue.

        Args:
            request_data (dict): Data to send to worker.
            timeout (int): Timeout in seconds.

        Returns:
            dict: Inference result.
        """
        self._request_queue.put(request_data)
        try:
            response = self._response_queue.get(timeout=timeout)
            if response['error']:
                logger.error(f"Backend inference error: {response['error']}")
                return {'answer': f"Error: {response['error']}", 'time': 'error'}
            return response['result']
        except mp.TimeoutError:
            logger.warning(f"Inference timed out after {timeout} seconds.")
            self._cleanup_queues()
            return {'answer': f'Request timed out after {timeout} seconds', 'time': f'{timeout}+ seconds'}
        except Exception as e:
            logger.error(f"Queue error during inference: {e}")
            return {'answer': f'Queue error: {str(e)}', 'time': 'error'}

    def _cleanup_queues(self) -> None:
        """Cleanup queues after timeout or error."""
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except:
                break
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except:
                break

    @staticmethod
    def convert_resize_image(image_array: np.ndarray, target_size: tuple[int, int] = (336, 336)) -> np.ndarray:
        """
        Convert and resize image for VLM using central crop to maintain aspect ratio.

        Args:
            image_array (np.ndarray): Input image (BGR).
            target_size (tuple[int, int], optional): Target size (width, height). Defaults to (336, 336).

        Returns:
            np.ndarray: Resized RGB image.
        """
        # Convert BGR to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        h, w = image_array.shape[:2]
        target_w, target_h = target_size

        # Scale to cover the target size (Central Crop strategy)
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image
        resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop
        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2
        cropped = resized[y_start:y_start+target_h, x_start:x_start+target_w]

        return cropped.astype(np.uint8)

    def close(self) -> None:
        """Close the backend process."""
        try:
            logger.info("Closing VLM backend...")
            self._request_queue.put(None)
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
            logger.info("VLM backend closed.")
        except Exception as e:
            logger.error(f"Error closing backend: {e}")
