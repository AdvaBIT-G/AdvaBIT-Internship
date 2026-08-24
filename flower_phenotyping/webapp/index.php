<?php
error_reporting(E_ALL);
ini_set('display_errors', 1);
require_once __DIR__ . "/config.php";
 
$python_bin  = "/home/gmartinez/miniconda3/envs/ultra_env/bin/python";
$script_path = "/home/gmartinez/AdvaBIT-Internship/flower_phenotyping/src/20260821_single_predict_pipeline.py";
 
$result = null;   // where the results from the models are saved to show later
$error = null;
 
// ------------------------------------------------------------
// IMAGE UPLOAD
// ------------------------------------------------------------
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_FILES["image"])) {
 
    $file = $_FILES["image"];
 
    if ($file["error"] != 0) {
        $error = "Upload error code: " . $file["error"];;
    } else {
 
        $extension = strtolower(pathinfo($file["name"], PATHINFO_EXTENSION));
        $allowed = ["jpg", "jpeg", "png"];
 
        if (!in_array($extension, $allowed)) {
            $error = "Only JPG or PNG extensions are allowed.";
        } else {
 
            $destination_folder = "uploads/";
            if (!is_dir($destination_folder)) {
                mkdir($destination_folder);
            }
 
            $filename = uniqid() . "." . $extension;
            $destination_path = $destination_folder . $filename;
 
            move_uploaded_file($file["tmp_name"], $destination_path);
 
            // --- Execute python script to get the predictions ---
            $command = escapeshellcmd($python_bin) . " " .
                       escapeshellarg($script_path) . " " .
                       escapeshellarg($destination_path);
 
            $output = shell_exec($command . " 2>&1");

            $data = json_decode($output, true);
 
            if ($data === null) {
                $error = "Could not read the prediction result.";
            } elseif (!empty($data["error"])) {
                $error = $data["error"];
            } else {
                
 
                // --- Save the result in a database ---
                $connection = connectDB();
 
                $stmt = $connection->prepare(
                    "INSERT INTO images (image_path, processed_at) VALUES (?, NOW())"
                );
                $stmt->bind_param("s", $destination_path);
                $stmt->execute();
                $id_image = $connection->insert_id;
                $stmt->close();
 
                if ($data["svm_class"] !== null) {
                    $stmt = $connection->prepare(
                        "INSERT INTO predictions (image_id, model_name, predicted_class, probability, created_at)
                         VALUES (?, 'svm', ?, ?, NOW())"
                    );
                    $stmt->bind_param("isd", $id_image, $data["svm_class"], $data["svm_probability"]);
                    $stmt->execute();
                    $stmt->close();
                }
 
                if ($data["lr_class"] !== null) {
                    $stmt = $connection->prepare(
                        "INSERT INTO predictions (image_id, model_name, predicted_class, probability, created_at)
                         VALUES (?, 'logistic_regression', ?, ?, NOW())"
                    );
                    $stmt->bind_param("isd", $id_image, $data["lr_class"], $data["lr_probability"]);
                    $stmt->execute();
                    $stmt->close();
                }
 
                // ------------------------------------------------------------
                // READ RESULTS FROM THE DATABASE
                // ------------------------------------------------------------
                $stmt = $connection->prepare(
                    "SELECT i.image_path, p.model_name, p.predicted_class, p.probability
                     FROM images i
                     JOIN predictions p ON p.image_id = i.id
                     WHERE i.id = ?"
                );
                $stmt->bind_param("i", $id_image);
                $stmt->execute();
                $rows = $stmt->get_result();
 
                $result = [
                    "image_path" => $destination_path,
                    "predictions" => [],
                ];
 
                while ($row = $rows->fetch_assoc()) {
                    $result["predictions"][] = $row;
                }
 
                $stmt->close();
                $connection->close();
            }
        }
    }
}
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FLOWER classifier</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
 
    <div class="container">
        <h1>FLOWER CLASSIFIER</h1>
        <p>Please upload a flower picture to get the color class</p>
 
        <form action="index.php" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
            <button type="submit">Analyze image</button>
        </form>
 
        <?php if ($error): ?>
            <p class="error"><?= htmlspecialchars($error) ?></p>
        <?php endif; ?>
 
        <?php if ($result): ?>
            <div class="result">
                <img src="<?= htmlspecialchars($result["image_path"]) ?>" alt="Uploaded image">
 
                <h2>Result</h2>
 
                <?php foreach ($result["predictions"] as $p): ?>
                    <p>
                        <strong><?= htmlspecialchars($p["model_name"]) ?>:</strong>
                        <?= htmlspecialchars($p["predicted_class"]) ?>
                        (<?= round($p["probability"] * 100, 1) ?>% of probability)
                    </p>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
 
    </div>
 
</body>
</html>