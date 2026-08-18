CREATE DATABASE IF NOT EXISTS flower_phenotyping
 
USE flower_phenotyping;
 
CREATE TABLE IF NOT EXISTS images (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    image_path      VARCHAR(500)   NOT NULL,
    processed_at    DATETIME       NOT NULL,
 
    INDEX idx_image_path (image_path),
    INDEX idx_processed_at (processed_at)
) ENGINE=InnoDB;
 
CREATE TABLE IF NOT EXISTS predictions (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    image_id            INT            NOT NULL,
    model_name          ENUM('svm', 'logistic_regression')  NOT NULL,
    predicted_class     VARCHAR(100)   NOT NULL,   
    probability         FLOAT          NOT NULL,
    created_at          DATETIME       NOT NULL,
 
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    INDEX idx_image_id (image_id),
    INDEX idx_model_name (model_name),
    UNIQUE KEY uq_image_model (image_id, model_name)
) ENGINE=InnoDB;
 