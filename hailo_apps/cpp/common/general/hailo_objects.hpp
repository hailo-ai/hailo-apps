/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
/**
 * @file hailo_objects.hpp
 * @authors Hailo
 **/

#pragma once

#include "hailo_tensors.hpp"
#include <map>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <mutex>

#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define CLIP(x) (CLAMP(x, 0, 255))
#define NULL_CLASS_ID (-1)

typedef enum
{
    HAILO_ROI,
    HAILO_CLASSIFICATION,
    HAILO_DETECTION,
    HAILO_LANDMARKS,
    HAILO_TILE,
    HAILO_UNIQUE_ID,
    HAILO_MATRIX,
    HAILO_DEPTH_MASK,
    HAILO_CLASS_MASK,
    HAILO_CONF_CLASS_MASK,
    HAILO_USER_META
} hailo_object_t;

static std::map<std::string, hailo_object_t> hailo_object_map = {
    {"hailo_roi", HAILO_ROI},
    {"hailo_classification", HAILO_CLASSIFICATION},
    {"hailo_detection", HAILO_DETECTION},
    {"hailo_landmarks", HAILO_LANDMARKS},
    {"hailo_tile", HAILO_TILE},
    {"hailo_unique_id", HAILO_UNIQUE_ID},
    {"hailo_matrix", HAILO_MATRIX},
    {"hailo_depth_mask", HAILO_DEPTH_MASK},
    {"hailo_class_mask", HAILO_CLASS_MASK},
    {"hailo_conf_class_mask", HAILO_CONF_CLASS_MASK},
    {"hailo_user_meta", HAILO_USER_META},
};

inline hailo_object_t hailo_object_type_from_string(const std::string &lower_case_str)
{
    if (hailo_object_map.find(lower_case_str) != hailo_object_map.end())
    {
        return hailo_object_map[lower_case_str];
    }
    else
    {
        throw std::invalid_argument("Invalid hailo object type: " + lower_case_str + ".");
    }
}

inline std::string hailo_object_type_to_string(const hailo_object_t obj)
{
    for (auto it = hailo_object_map.begin(); it != hailo_object_map.end(); ++it)
    {
        if (it->second == obj)
        {
            return it->first;
        }
    }
    throw std::invalid_argument("Invalid hailo object type.");
}
typedef enum
{
    SINGLE_SCALE,
    MULTI_SCALE,
} hailo_tiling_mode_t;

typedef enum
{
    TRACKING_ID,
    GLOBAL_ID,
} hailo_unique_id_mode_t;

static float assure_normal(float num)
{
    if ((num > 1.0f) || (num < 0.0))
    {
        throw std::invalid_argument("Number should be between 0.0 to 1.0.");
    }
    return num;
}

/**
 * @brief HailoPoint - Represents a detected point (Landmarks)
 *
 */
struct HailoPoint
{
protected:
    const float m_x;
    const float m_y;
    const float m_confidence;

public:
    /**
     * @brief Construct a new Hailo Point object
     *
     * @param x normalized x position (float)
     * @param y normalized y position (float)
     * @param confidence The confidence in the point's accuracy, float between 0.0 to 1.0 - default is 1.0.
     */
    HailoPoint(float x, float y, float confidence = 1.0f) : m_x(x), m_y(y), m_confidence(assure_normal(confidence)){};
    const float x() const { return m_x; }
    const float y() const { return m_y; }
    const float confidence() const { return m_confidence; }
};

/**
 * @brief HailoBBox - Represents a bounding box.
 * Takes 4 float arguments representing the bounding box (normalized).
 * The first 2 arguments are the x and y of the minimum point (Top Left corner).
 * The other 2 arguments are the width and height of the box respectivly.
 * All arguments are normalized to the picture they relate to.
 */
struct HailoBBox
{
protected:
    float m_xmin;
    float m_ymin;
    float m_width;
    float m_height;

public:
    /**
     * @brief Construct a new Hailo BBox object
     *
     * @param xmin normalized xmin position
     * @param ymin normalized ymin position
     * @param width normalized width of bounding box
     * @param height normalized height of bounding box
     */
    HailoBBox(float xmin, float ymin, float width, float height) : m_xmin(xmin), m_ymin(ymin), m_width(width), m_height(height){};

    const float xmin() const { return m_xmin; }
    const float ymin() const { return m_ymin; }
    const float width() const { return m_width; }
    const float height() const { return m_height; }
    const float xmax() const { return m_xmin + m_width; }
    const float ymax() const { return m_ymin + m_height; }
};

/**
 * @brief Represents an object that is a usable output after postprocessing.
 * An abstract class for all objects to inherit from.
 */
class HailoObject
{
protected:
    std::shared_ptr<std::mutex> mutex;

public:
    // Constructor
    HailoObject()
    {
        mutex = std::make_shared<std::mutex>();
    };
    // Destructor
    virtual ~HailoObject() = default;
    HailoObject &operator=(const HailoObject &other) = default;
    HailoObject &operator=(HailoObject &&other) noexcept = default;
    HailoObject(HailoObject &&other) noexcept = default;
    HailoObject(const HailoObject &other) = default;

    /**
     * @brief Get the type object
     *
     * @return hailo_object_t - The type of the object.
     */
    virtual hailo_object_t get_type() = 0;
};

using HailoObjectPtr = std::shared_ptr<HailoObject>;

/**
 * @brief Represents a HailoObject that can hold other objects.
 *  for example a face detection can hold landmarks or age classification, gender classification etc...
 */
class HailoMainObject : public HailoObject, public std::enable_shared_from_this<HailoMainObject>
{
protected:
    std::vector<HailoObjectPtr> m_sub_objects;
    std::map<std::string, HailoTensorPtr> m_tensors;

public:
    HailoMainObject()
    {
        mutex = std::make_shared<std::mutex>();
    };
    virtual ~HailoMainObject() = default;
    HailoMainObject(HailoMainObject &&other) noexcept : HailoObject(other), m_sub_objects(std::move(other.m_sub_objects)){};
    HailoMainObject(const HailoMainObject &other) : HailoObject(other), m_sub_objects(other.m_sub_objects){};
    HailoMainObject &operator=(const HailoMainObject &other) = default;
    HailoMainObject &operator=(HailoMainObject &&other) noexcept = default;

    /**
     * @brief Add an object to the main object.
     *
     * @param obj Object to add.
     */
    void add_object(HailoObjectPtr obj)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.emplace_back(obj);
    };

    /**
     * @brief Add a tensor to the main object.
     *
     * @param tensor Tensor to add.
     */
    void add_tensor(HailoTensorPtr tensor)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_tensors.emplace(tensor->name(), tensor);
    };

    /**
     * @brief Remove a HailoObject from the MainObject
     *
     * @param obj  -  HailoObjectPtr
     *        The object to remove
     */
    void remove_object(HailoObjectPtr obj)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.erase(std::remove(m_sub_objects.begin(), m_sub_objects.end(), obj), m_sub_objects.end());
    };

    /**
     * @brief Remove a HailoObject from the MainObject
     *
     * @param index  -  uint
     *        The index of the object to remove
     */
    void remove_object(uint index)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_sub_objects.erase(m_sub_objects.begin() + index);
    };

    /**
     * @brief Get a tensor from this main object.
     *
     * @param name Tensor's name to get,
     * @return HailoTensorPtr - A tensor.
     */
    HailoTensorPtr get_tensor(std::string name)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        auto itr = m_tensors.find(name);
        if (itr == m_tensors.end())
        {
            throw std::invalid_argument("No tensor with name " + name);
        }
        return itr->second;
    };

    /**
     * @brief Checks whether there are tensors attached to this main object
     *
     * @return true when there are tensors in this main object.
     * @return false when there are no tensors in this main object.
     */
    bool has_tensors()
    {
        return !m_tensors.empty();
    };

    /**
     * @brief Get a vector of the tensors attached to this main object.
     *
     * @return std::vector<HailoTensorPtr>
     */
    std::vector<HailoTensorPtr> get_tensors()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<HailoTensorPtr> _tensors;
        _tensors.reserve(m_tensors.size());
        for (auto &tensor_pair : m_tensors)
        {
            _tensors.emplace_back(tensor_pair.second);
        }
        return _tensors;
    };

    std::map<std::string, HailoTensorPtr> get_tensors_by_name()
    {
        std::map<std::string, HailoTensorPtr> tensors_by_name;
        auto tensors = get_tensors();
        for (auto tensor : tensors)
        {
            tensors_by_name[tensor->name()] = tensor;
        }
        return tensors_by_name;
    }

    /**
     * @brief Clear all tensors attached to this main object.
     *
     */
    void clear_tensors()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_tensors.clear();
    }

    /**
     * @brief Get the objects attached to this main object
     *
     * @return std::vector<HailoObjectPtr>
     */
    std::vector<HailoObjectPtr> get_objects()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_sub_objects;
    }

    /**
     * @brief Get the objects of a given type, attached to this main object.
     *
     * @param type The type of object to get.
     * @return std::vector<HailoObjectPtr>
     */
    std::vector<HailoObjectPtr> get_objects_typed(hailo_object_t type)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        std::vector<HailoObjectPtr> filtered_subobjects;
        for (auto &obj : m_sub_objects)
        {
            if (obj->get_type() == type)
            {
                filtered_subobjects.emplace_back(obj);
            }
        }
        return filtered_subobjects;
    }

    /**
     * @brief Removes all the objects of a given type, attached to this main object.
     *
     * @param type The type of object to get.
     */
    void remove_objects_typed(hailo_object_t type)
    {
        for (auto obj : this->get_objects_typed(type))
        {
            this->remove_object(obj);
        }
    }
};
using HailoMainObjectPtr = std::shared_ptr<HailoMainObject>;

/**
 * @brief Represents an ROI (Region Of Interest).
 * Part of an image, can hold other objects. Mostly inherited by other objects but isn't abstract.
 * Can represents the whole image by giving the right HailoBBox.
 */
class HailoROI : public HailoMainObject
{
protected:
    HailoBBox m_bbox;         // A bounding box - the normalized position of this region of interest.
    HailoBBox m_scaling_bbox; // A bounding box to scale by - x offset, y offset, width factor, height factor
    std::string m_stream_id;  // A string representing the stream ID that is related to this ROI.
public:
    HailoROI(HailoBBox bbox, std::string stream_id = "") : m_bbox(bbox), m_scaling_bbox(HailoBBox(0.0, 0.0, 1.0, 1.0)), m_stream_id(stream_id){};
    virtual ~HailoROI() = default;
    HailoROI(HailoROI &&other) noexcept : HailoMainObject(other), m_bbox(std::move(other.m_bbox)), m_scaling_bbox(std::move(other.m_scaling_bbox)), m_stream_id(std::move(other.m_stream_id)){};
    HailoROI(const HailoROI &other) : HailoMainObject(other), m_bbox(other.m_bbox), m_scaling_bbox(std::move(other.m_scaling_bbox)), m_stream_id(std::move(other.m_stream_id)){};
    HailoROI &operator=(const HailoROI &other) = default;
    HailoROI &operator=(HailoROI &&other) noexcept = default;
    std::shared_ptr<HailoROI> shared_from_this()
    {
        return std::dynamic_pointer_cast<HailoROI>(HailoMainObject::shared_from_this());
    }
    virtual hailo_object_t get_type()
    {
        return HAILO_ROI;
    }

    /**
     * @brief Add an object to the main object.
     *
     * @param obj Object to add.
     */
    void add_object(HailoObjectPtr obj)
    {
        std::shared_ptr<HailoROI> possible_roi = std::dynamic_pointer_cast<HailoROI>(obj);
        if (nullptr != possible_roi)
        {
            possible_roi->set_scaling_bbox(this->get_bbox());
            possible_roi->set_stream_id(this->get_stream_id());
        }
        HailoMainObject::add_object(obj);
    };

    /**
     * @brief Add an object to the main object.
     *        Ignore possible scaling of rois
     *
     * @param obj Object to add.
     */
    void add_unscaled_object(HailoObjectPtr obj)
    {
        HailoMainObject::add_object(obj);
    };

    /**
     * @brief Get the bbox of this ROI
     *
     * @return HailoBBox&
     */
    HailoBBox &get_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_bbox;
    }

    /**
     * @brief Set the bbox of this ROI
     *
     * @param bbox
     */
    void set_bbox(HailoBBox bbox)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_bbox = std::move(bbox);
    }

    /**
     * @brief Get the scaling bbox of this ROI
     *
     * @return HailoBBox&
     */
    HailoBBox &get_scaling_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_scaling_bbox;
    }

    /**
     * @brief Set the scaling bbox of this ROI
     *
     * @param bbox
     */
    void set_scaling_bbox(HailoBBox bbox)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        float new_xmin = (m_scaling_bbox.xmin() * bbox.width()) + bbox.xmin();
        float new_ymin = (m_scaling_bbox.ymin() * bbox.height()) + bbox.ymin();
        float new_width = m_scaling_bbox.width() * bbox.width();
        float new_height = m_scaling_bbox.height() * bbox.height();
        HailoBBox new_scale = HailoBBox(new_xmin, new_ymin, new_width, new_height);
        m_scaling_bbox = std::move(new_scale);
    }

    /**
     * @brief Clear the scaling bbox of this ROI
     *
     */
    void clear_scaling_bbox()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_scaling_bbox = HailoBBox(0.0, 0.0, 1.0, 1.0);
    }

    /**
     * @brief Get the stream ID of this ROI
     *
     * @return std::string
     */
    std::string get_stream_id()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_stream_id;
    }

    /**
     * @brief Set the stream ID of this ROI
     *
     * @param stream_id
     */
    void set_stream_id(std::string stream_id)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_stream_id = std::move(stream_id);
    }
};
using HailoROIPtr = std::shared_ptr<HailoROI>;

class HailoTileROI : public HailoROI
{
protected:
    uint m_index;
    float m_overlap_x_axis;
    float m_overlap_y_axis;
    uint m_layer;
    hailo_tiling_mode_t m_mode;

public:
    HailoTileROI(HailoBBox bbox, uint index, float overlap_x_axis, float overlap_y_axis, uint layer, hailo_tiling_mode_t mode) : HailoROI(bbox), m_index(index), m_overlap_x_axis(overlap_x_axis), m_overlap_y_axis(overlap_y_axis), m_layer(layer), m_mode(mode){};
    // Move constructor
    HailoTileROI(HailoTileROI &&other) noexcept : HailoROI(other),
                                                  m_index(other.m_index),
                                                  m_overlap_x_axis(other.m_overlap_x_axis),
                                                  m_overlap_y_axis(other.m_overlap_y_axis),
                                                  m_layer(other.m_layer),
                                                  m_mode(other.m_mode){};
    // Copy constructor
    HailoTileROI(const HailoTileROI &other) : HailoROI(other),
                                              m_index(other.m_index),
                                              m_overlap_x_axis(other.m_overlap_x_axis),
                                              m_overlap_y_axis(other.m_overlap_y_axis),
                                              m_layer(other.m_layer),
                                              m_mode(other.m_mode){};
    // Move assignment
    HailoTileROI &operator=(HailoTileROI &&other) noexcept
    {
        if (this != &other)
        {
            m_bbox = std::move(other.m_bbox);
            m_sub_objects = std::move(other.m_sub_objects);
            m_index = other.m_index;
            m_overlap_x_axis = other.m_overlap_x_axis;
            m_overlap_y_axis = other.m_overlap_y_axis;
            m_layer = other.m_layer;
            m_mode = other.m_mode;
        }
        return *this;
    };
    // Copy assignment
    HailoTileROI &operator=(const HailoTileROI &other)
    {
        if (this != &other)
        {
            m_bbox = other.m_bbox;
            m_sub_objects = other.m_sub_objects;
            m_index = other.m_index;
            m_overlap_x_axis = other.m_overlap_x_axis;
            m_overlap_y_axis = other.m_overlap_y_axis;
            m_layer = other.m_layer;
            m_mode = other.m_mode;
        }
        return *this;
    };
    virtual ~HailoTileROI() = default;
    virtual hailo_object_t get_type()
    {
        return HAILO_TILE;
    }

    float get_overlap_x_axis() { return m_overlap_x_axis; }
    float get_overlap_y_axis() { return m_overlap_y_axis; }
    uint get_index() { return m_index; }
    uint get_layer() { return m_layer; }
    uint get_mode() { return m_mode; }
};
using HailoTileROIPtr = std::shared_ptr<HailoTileROI>;

/**
 * @brief Represents a detection in a ROI. Inherits from HailoROI.
 *
 */
class HailoDetection : public HailoROI
{
protected:
    float m_confidence;  // Confidence of the detection.
    std::string m_label; // The label of detection, e.g. "Horse", "Monkey", "Tiger" for type "Animals".
    int m_class_id;      // Class id, initialized to -1 if missing.
public:
    /**
     * @brief Construct a new New Hailo Detection object
     *
     * @param bbox HailoBBox - a bounding box representing the region of interest in the frame.
     * @param label std::string what the detection is.
     * @param confidence The confidence of the detection.
     * @note class id is set to -1.
     */
    HailoDetection(HailoBBox bbox, const std::string &label, float confidence) : HailoROI(bbox), m_confidence(assure_normal(confidence)), m_label(label), m_class_id(NULL_CLASS_ID){};
    /**
     * @brief Construct a new New Hailo Detection object
     *
     * @param bbox HailoBBox - a bounding box representing the region of interest in the frame.
     * @param class_id The detection's class id, if theres any.
     * @param label std::string what the detection is.
     * @param confidence The confidence of the detection.
     */
    HailoDetection(HailoBBox bbox, int class_id, const std::string &label, float confidence) : HailoROI(bbox), m_confidence(assure_normal(confidence)), m_label(label), m_class_id(class_id){};

    // Move constructor
    HailoDetection(HailoDetection &&other) noexcept : HailoROI(other),
                                                      m_confidence(assure_normal(other.m_confidence)),
                                                      m_label(std::move(other.m_label)),
                                                      m_class_id(other.m_class_id){};
    // Copy constructor
    HailoDetection(const HailoDetection &other) : HailoROI(other),
                                                  m_confidence(assure_normal(other.m_confidence)),
                                                  m_label(other.m_label),
                                                  m_class_id(other.m_class_id){};
    virtual ~HailoDetection() = default;

    // Move assignment
    HailoDetection &operator=(HailoDetection &&other) noexcept
    {
        if (this != &other)
        {
            HailoROI::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_class_id = other.m_class_id;
            m_label = std::move(other.m_label);
        }
        return *this;
    };
    // Copy assignment
    HailoDetection &operator=(const HailoDetection &other)
    {
        if (this != &other)
        {
            HailoROI::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_class_id = other.m_class_id;
            m_label = other.m_label;
        }
        return *this;
    };
    // Overload comparison operators
    bool operator<(const HailoDetection &other) const
    {
        return this->m_confidence < other.m_confidence;
    }

    bool operator>(const HailoDetection &other) const
    {
        return this->m_confidence > other.m_confidence;
    }

    virtual hailo_object_t get_type()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return HAILO_DETECTION;
    }

    std::shared_ptr<HailoObject> clone()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return std::make_shared<HailoDetection>(*this);
    }

    // Getters of DetectionObject.

    float get_confidence()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_confidence;
    }
    void set_confidence(float conf)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_confidence = conf;
    }
    std::string get_label()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_label;
    }
    void set_label(std::string label)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_label = label;
    }
    int get_class_id()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_class_id;
    }
};
using HailoDetectionPtr = std::shared_ptr<HailoDetection>;

/**
 * @brief Represents a Classification of an ROI.
 *
 */
class HailoClassification : public HailoObject
{
protected:
    float m_confidence;                // Confidence of the classification.
    std::string m_classification_type; // Type of labeling, e.g. "age", "gender", "color", etc...
    std::string m_label;               // The label of classification, e.g. "Horse", "Monkey", "Tiger" for type "Animals".
    int m_class_id;                    // Class id, initialized to -1 if missing.
public:
    /**
     * @brief Construct a new Hailo Classification object
     *
     * @param classification_type The type of classification.
     * @param label classification result.
     * @param confidence confidence of classification result.
     */
    HailoClassification(const std::string &classification_type,
                        const std::string &label,
                        float confidence) : m_confidence(assure_normal(confidence)), m_classification_type(classification_type), m_label(label), m_class_id(NULL_CLASS_ID){};

    /**
     * @brief Construct a new Hailo Classification object
     *
     * @param classification_type The type of classification.
     * @param label classification result.
     */
    HailoClassification(const std::string &classification_type,
                        const std::string &label) : m_confidence(assure_normal(1.0f)), m_classification_type(classification_type), m_label(label), m_class_id(NULL_CLASS_ID){};

    /**
     * @brief Construct a new Hailo Classification object
     *
     * @param classification_type The type of classification.
     * @param class_id Class ID of the classification result.
     * @param label classification result.
     * @param confidence confidence of classification result.
     */
    HailoClassification(const std::string &classification_type,
                        int class_id,
                        std::string label,
                        float confidence) : m_confidence(assure_normal(confidence)), m_classification_type(classification_type), m_label(label), m_class_id(class_id){};
    // Move Constructor
    HailoClassification(HailoClassification &&other) : m_confidence(assure_normal(other.m_confidence)),
                                                       m_classification_type(std::move(other.m_classification_type)),
                                                       m_label(std::move(other.m_label)), m_class_id(other.m_class_id){};
    // Copy Constructor
    HailoClassification(const HailoClassification &other) : m_confidence(assure_normal(other.m_confidence)),
                                                            m_classification_type(other.m_classification_type),
                                                            m_label(other.m_label), m_class_id(other.m_class_id){};
    virtual ~HailoClassification() = default;
    // Move assignment
    HailoClassification &operator=(HailoClassification &&other) noexcept
    {
        if (this != &other)
        {
            HailoObject::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_classification_type = std::move(other.m_classification_type);
            m_label = std::move(other.m_label);
            m_class_id = other.m_class_id;
        }
        return *this;
    };
    // Copy assignment
    HailoClassification &operator=(const HailoClassification &other)
    {
        if (this != &other)
        {
            HailoObject::operator=(other);
            m_confidence = assure_normal(other.m_confidence);
            m_classification_type = other.m_classification_type;
            m_label = other.m_label;
            m_class_id = other.m_class_id;
        }
        return *this;
    };

    std::shared_ptr<HailoObject> clone()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return std::make_shared<HailoClassification>(*this);
    }

    virtual hailo_object_t get_type()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return HAILO_CLASSIFICATION;
    }

    // Getters to HailoClassification

    float get_confidence()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_confidence;
    }
    std::string get_label()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_label;
    }
    std::string get_classification_type()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_classification_type;
    }
    int get_class_id()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_class_id;
    }
};
using HailoClassificationPtr = std::shared_ptr<HailoClassification>;

/**
 * @brief Represents a set of landmarks on a given ROI.
 *
 */
class HailoLandmarks : public HailoObject
{
protected:
    std::string m_landmarks_type;                   // Type of labeling, e.g. "pose", "facial landmarking", etc...
    std::vector<HailoPoint> m_points;               // Vector of points.
    float m_threshold;                              // Threshold of landmark network.
    const std::vector<std::pair<int, int>> m_pairs; // pairs of landmarks that should be connected in the overlay
public:
    /**
     * @brief Construct a new Hailo Landmarks object
     *
     * @param landmarks_name landmarks_name Type of landmarks.
     * @param threshold threshold Minimum threshold of points to decide whether they are valid.
     * @param pairs vector of pairs of joints that should be connected in overlay
     */
    HailoLandmarks(std::string landmarks_name, float threshold = 0.0f, const std::vector<std::pair<int, int>> pairs = {}) : m_landmarks_type(landmarks_name), m_threshold(threshold), m_pairs(pairs){};
    /**
     * @brief Construct a new Hailo Landmarks object
     *
     * @param landmarks_name Type of landmarks.
     * @param points Set of landmarks represented as std::vector<HailoPoint>.
     * @param threshold Minimum threshold of points to decide whether they are valid.
     */
    HailoLandmarks(std::string landmarks_name,
                   std::vector<HailoPoint> points,
                   float threshold = 0.0f,
                   const std::vector<std::pair<int, int>> pairs = {}) : m_landmarks_type(landmarks_name),
                                                                        m_points(std::move(points)),
                                                                        m_threshold(threshold),
                                                                        m_pairs(pairs){};
    virtual ~HailoLandmarks() = default;
    // Move constructor
    HailoLandmarks(HailoLandmarks &&other) : m_landmarks_type(std::move(other.m_landmarks_type)), m_points(std::move(other.m_points)), m_threshold(other.m_threshold), m_pairs(other.m_pairs){};
    // Copy constructor
    HailoLandmarks(const HailoLandmarks &other) : m_landmarks_type(other.m_landmarks_type), m_points(other.m_points), m_threshold(other.m_threshold), m_pairs(other.m_pairs){};
    HailoLandmarks &operator=(const HailoLandmarks &other) = default;
    HailoLandmarks &operator=(HailoLandmarks &&other) noexcept = default;

    virtual hailo_object_t get_type()
    {
        return HAILO_LANDMARKS;
    }

    /**
     * @brief Add a point to this Landmarks object.
     *
     * @param point HailoPoint to add.
     */
    void add_point(HailoPoint point)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_points.emplace_back(point);
    };

    /**
     * @brief Set a new vector of points to this Landmarks object.
     *
     * @param points std::vector<HailoPoint> to set.
     */
    void set_points(std::vector<HailoPoint> points)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_points.clear();
        m_points = std::move(points);
    };

    std::shared_ptr<HailoObject> clone()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return std::make_shared<HailoLandmarks>(*this);
    }

    // Getters for HailoLandmarks object.

    std::vector<HailoPoint> get_points()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_points;
    }
    float get_threshold()
    {
        return m_threshold;
    }
    std::string get_landmarks_type()
    {
        return m_landmarks_type;
    }
    std::vector<std::pair<int, int>> get_pairs()
    {
        return m_pairs;
    }
};
using HailoLandmarksPtr = std::shared_ptr<HailoLandmarks>;

/**
 * @brief Represents a unique id of a ROI.
 *
 */
class HailoUniqueID : public HailoObject
{
protected:
    int m_unique_id;               // Unique id, initialized to -1 if missing.
    hailo_unique_id_mode_t m_mode; // Mode of unique id.

public:
    /**
     * @brief Construct a new Hailo Unique ID object
     *
     * @param unique_id  -  int
     *        A unique id
     */
    HailoUniqueID(int unique_id, hailo_unique_id_mode_t mode = TRACKING_ID) : m_unique_id(unique_id), m_mode(mode){};

    virtual ~HailoUniqueID() = default;

    int get_id()
    {
        return m_unique_id;
    }
    int get_mode()
    {
        return m_mode;
    }

    std::shared_ptr<HailoObject> clone()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return std::make_shared<HailoUniqueID>(*this);
    }

    virtual hailo_object_t get_type()
    {
        return HAILO_UNIQUE_ID;
    }
};
using HailoUniqueIDPtr = std::shared_ptr<HailoUniqueID>;

class HailoMask : public HailoObject
{
protected:
    int m_mask_width;
    int m_mask_height;
    float m_transparency;

public:
    HailoMask(int mask_width, int mask_height, float transparency) : m_mask_width(mask_width), m_mask_height(mask_height), m_transparency(transparency){};

    // Move Constructor
    HailoMask(HailoMask &&other) = default;
    // Copy Constructor
    HailoMask(const HailoMask &other) = default;
    virtual ~HailoMask() = default;
    // Move assignment
    HailoMask &operator=(HailoMask &&other) noexcept = default;

    // Copy assignment
    HailoMask &operator=(const HailoMask &other) = default;

    int get_width()
    {
        return m_mask_width;
    }

    int get_height()
    {
        return m_mask_height;
    }

    float get_transparency()
    {
        return m_transparency;
    }
};
using HailoMaskPtr = std::shared_ptr<HailoMask>;

class HailoDepthMask : public HailoMask
{
protected:
    std::vector<float> m_data;

public:
    HailoDepthMask(std::vector<float> &&data_vec, int mask_width, int mask_height, float transparency) : HailoMask(mask_width, mask_height, transparency), m_data(std::move(data_vec)){};

    virtual hailo_object_t get_type()
    {
        return HAILO_DEPTH_MASK;
    }

    const std::vector<float> &get_data()
    {
        return m_data;
    }
    virtual ~HailoDepthMask() = default;
};
using HailoDepthMaskPtr = std::shared_ptr<HailoDepthMask>;

class HailoClassMask : public HailoMask
{
protected:
    std::vector<uint8_t> m_data;

public:
    HailoClassMask(std::vector<uint8_t> &&data_vec, int mask_width, int mask_height, float transparency) : HailoMask(mask_width, mask_height, transparency), m_data(std::move(data_vec)){};

    virtual hailo_object_t get_type()
    {
        return HAILO_CLASS_MASK;
    }

    const std::vector<uint8_t> &get_data()
    {
        return m_data;
    }
    virtual ~HailoClassMask() = default;
};
using HailoClassMaskPtr = std::shared_ptr<HailoClassMask>;

class HailoConfClassMask : public HailoMask
{
protected:
    int m_class_id;
    std::vector<float> m_data;

public:
    HailoConfClassMask(std::vector<float> &&data_vec, int mask_width, int mask_height, float transparency, int class_id) : HailoMask(mask_width, mask_height, transparency), m_class_id(class_id), m_data(std::move(data_vec)){};

    virtual hailo_object_t get_type()
    {
        return HAILO_CONF_CLASS_MASK;
    }

    const std::vector<float> &get_data()
    {
        return m_data;
    }

    virtual ~HailoConfClassMask() = default;

    int get_class_id()
    {
        return m_class_id;
    }
};
using HailoConfClassMaskPtr = std::shared_ptr<HailoConfClassMask>;

class HailoMatrix : public HailoObject
{
protected:
    std::vector<float> m_data;
    uint32_t m_mat_height;
    uint32_t m_mat_width;
    uint32_t m_mat_features;

public:
    static const int DEFAULT_NUMBER_OF_FEATURES = 1;
    HailoMatrix(std::vector<float> data,
                uint32_t mat_height,
                uint32_t mat_width,
                uint32_t mat_features = HailoMatrix::DEFAULT_NUMBER_OF_FEATURES) : m_data(data),
                                                                                   m_mat_height(mat_height),
                                                                                   m_mat_width(mat_width),
                                                                                   m_mat_features(mat_features){};

    std::shared_ptr<HailoObject> clone()
    {
        return std::dynamic_pointer_cast<HailoObject>(std::make_shared<HailoMatrix>(*this));
    }

    const uint32_t width()
    {
        return m_mat_width;
    }
    const uint32_t height()
    {
        return m_mat_height;
    }
    const uint32_t features()
    {
        return m_mat_features;
    }
    const uint32_t size() const
    {
        return m_mat_width * m_mat_height * m_mat_features;
    }
    std::vector<std::size_t> shape()
    {
        std::vector<std::size_t> shape = {height(), width(), features()};
        return shape;
    }
    const std::vector<float> &get_data()
    {
        return m_data;
    }
    virtual hailo_object_t get_type()
    {
        return HAILO_MATRIX;
    }
};
using HailoMatrixPtr = std::shared_ptr<HailoMatrix>;

/**
 * @brief Represents a Sample metadata for users
 */
class HailoUserMeta : public HailoObject
{
protected:
    int m_user_int;
    std::string m_user_string;
    float m_user_float;

public:
    HailoUserMeta(){};
    HailoUserMeta(int user_int, std::string user_string, float user_float) : m_user_int(user_int), m_user_string(user_string), m_user_float(user_float){};

    virtual hailo_object_t get_type()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return HAILO_USER_META;
    }

    float get_user_float()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_user_float;
    }
    std::string get_user_string()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_user_string;
    }
    int get_user_int()
    {
        std::lock_guard<std::mutex> lock(*mutex);
        return m_user_int;
    }
    void set_user_float(float user_float)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_user_float = user_float;
    }
    void set_user_string(std::string user_string)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_user_string = user_string;
    }
    void set_user_int(int user_int)
    {
        std::lock_guard<std::mutex> lock(*mutex);
        m_user_int = user_int;
    }
};
using HailoUserMetaPtr = std::shared_ptr<HailoUserMeta>;