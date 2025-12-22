# region imports
# Standard library imports
from datetime import datetime, timedelta
from io import BytesIO

# Third-party library imports
from PIL import Image

# endregion imports


class TelegramHandler:
    def __init__(self, token, chat_id):
        """Initialize the TelegramHandler.
        Import telebot and set up the bot only if token and chat_id are provided.
        """
        self.bot = None
        self.chat_id = chat_id
        self.ids_msg_sent = {}  # Dictionary to store the last sent time for each person

        if token and chat_id:
            try:
                import telebot  # pyTelegramBotAPI

                self.bot = telebot.TeleBot(token)
            except ImportError:
                raise ImportError(
                    "The 'telebot' library is not installed. Install it using 'pip install pyTelegramBotAPI'."
                ) from None
        else:
            raise ValueError("Telegram token and chat ID must be provided.")

    def should_send_notification(self, global_id):
        """Check if a notification should be sent for the given global_id."""
        current_time = datetime.now()
        last_sent_time = self.ids_msg_sent.get(global_id)

        # Send notification if it has never been sent or if more than 1 hour has passed
        if last_sent_time is None or current_time - last_sent_time > timedelta(hours=1):
            self.ids_msg_sent[global_id] = current_time
            return True
        return False

    def send_notification(self, name, global_id, confidence, frame):
        """Send a notification via Telegram with the given details."""
        if not self.bot:
            raise ValueError("Telegram bot is not initialized. Provide a valid token and chat ID.")

        # Determine the caption for the notification
        if not name:
            caption = "🚨 Unknown person detected!"
        elif name == "Unknown":
            caption = f"Detected {global_id} (confidence: {confidence:.2f})"
        else:
            caption = f"Detected {name} (confidence: {confidence:.2f})"

        # Convert the frame to an image and send it
        image = Image.fromarray(frame)
        image_byte_array = BytesIO()
        image.save(image_byte_array, format="PNG")
        image_byte_array.seek(0)

        try:
            self.bot.send_photo(self.chat_id, image_byte_array, caption)
        except Exception as e:
            print(f"Error sending Telegram notification: {e!s}")
