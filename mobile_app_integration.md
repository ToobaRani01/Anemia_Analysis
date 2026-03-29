# 📦 Model Deployment Guide (Android + TensorFlow Lite)

---

## 🔍 Model Details

| Feature        | Value                          |
|----------------|--------------------------------|
| Model Name     | `anemia_model.tflite`          |
| Input Shape 1  | Image: `(1, 224, 224, 3)`      |
| Input Shape 2  | Meta: `(1, 2)` [Age, Gender]   |
| Output         | `(1, 1)` Normalized Hemoglobin |
| Framework      | TensorFlow Lite                |
| Task           | Regression & Classification    |

---

## ⚠️ Preprocessing Requirement (CRITICAL)

The model will only produce correct predictions if preprocessing exactly matches the training pipeline.

**Required Steps for Image:**
1. Resize image → 224 × 224
2. Convert to float
3. Normalize pixel values → 0 to 1

```java
// Pseudo-logic for image
image = image / 255.0f;
```

**Required Steps for Meta Input:**
1. Normalize Age → `Age / 100.0`
2. Encode Gender → Male = `1.0`, Female = `0.0`

❗ If preprocessing does not match:
- Incorrect hemoglobin predictions
- Inaccurate anemia clinical grading

---

## 🛠️ Implementation Guide (Android)

### 📌 STEP 1: Install Android Studio
- Download and install Android Studio.

### 📌 STEP 2: Create a New Project
1. Open Android Studio
2. Click **New Project**
3. Select **Empty Activity**
4. Choose **Java**
5. Click **Finish**

### 📌 STEP 3: Add TensorFlow Lite Dependencies
Open `build.gradle (Module: app)` and add:

```gradle
implementation 'org.tensorflow:tensorflow-lite:2.13.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
```

Click **Sync Now**

### 📌 STEP 4: Add Model File
- Navigate to: `app/src/main/`
- Create folder: `assets`
- Place model file: `anemia_model.tflite` inside the `assets` folder.

### 📌 STEP 5: Load Model in Android

```java
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import android.content.res.AssetFileDescriptor;

Interpreter tflite;

try {
    tflite = new Interpreter(loadModelFile());
} catch (Exception e) {
    e.printStackTrace();
}

private MappedByteBuffer loadModelFile() throws IOException {
    AssetFileDescriptor fileDescriptor = getAssets().openFd("anemia_model.tflite");
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
}
```

---

### 📌 STEP 6: Capture or Select Image
- Capture via camera 📷
- Select from gallery 🖼️

```java
Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);
```

---

### 📌 STEP 7: Convert Inputs to Tensors (Image + Meta)

```java
// 1. Prepare Image Input Tensor (1, 224, 224, 3)
float[][][][] imageInput = new float[1][224][224][3];

for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
        int pixel = resized.getPixel(x, y);
        // Normalize 0-255 to 0.0-1.0
        imageInput[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f; // Red
        imageInput[0][y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;  // Green
        imageInput[0][y][x][2] = (pixel & 0xFF) / 255.0f;         // Blue
    }
}

// 2. Prepare Meta Input Tensor (1, 2)
int patientAge = 25; 
String patientGender = "female"; 

float[][] metaInput = new float[1][2];
metaInput[0][0] = patientAge / 100.0f; // Normalized Age
metaInput[0][1] = patientGender.equalsIgnoreCase("male") ? 1.0f : 0.0f; // Gender Bin
```

---

### 📌 STEP 8: Run Prediction
The model requires both inputs (Image and Meta) to predict Hemoglobin.

```java
import java.util.HashMap;
import java.util.Map;

// Output array for hemoglobin
float[][] hgbOutput = new float[1][1];

Object[] inputs = new Object[]{ imageInput, metaInput };
Map<Integer, Object> outputs = new HashMap<>();
outputs.put(0, hgbOutput);

// Run model
tflite.runForMultipleInputsOutputs(inputs, outputs);

// Get the normalized prediction
float predictedNormalizedHgb = hgbOutput[0][0];
```

---

### 📌 STEP 9: Process Output (Hemoglobin, Anemia Status, Severity 0-3, & Risk Probability)

This step perfectly replicates the **entire** logic from the python project's root `src/utils.py` and `src/config.py` files. It translates the raw prediction, mathematically checks if the user is anemic, evaluates their severity level (0-3), and calculates a risk probability percentage.

#### 9a. Convert Normalized Hgb to Actual g/dL
The TFLite model gives us a raw fraction (between 0.0 and 1.0). We use the AI scaler values (Min 7.0, Max 17.4) to reverse this back into the actual Hemoglobin.

#### 9b. Load Dynamic Clinical Thresholds
You must check the patient's demographics to load their specific healthy baselines. Below are the **exact matrices** straight from our project's `config.py`. Program these mappings into the app so you can dynamically load the `minNormalHb` and the `mild`, `moderate`, and `severe` breakpoints.

**Normal Hemoglobin Baseline Matrix**
| Demographic | Age | Minimum Hb | Maximum Hb |
| :--- | :--- | :--- | :--- |
| Children (M/F) | 0–4 | 11.0 | 14.0 |
| Children (M/F) | 5–11 | 11.5 | 14.5 |
| Children (M/F) | 12–14 | 12.0 | 15.0 |
| Men | 15+ | 13.0 | 17.0 |
| Women (Non-Pregnant) | 15+ | 12.0 | 15.5 |
| Women (Pregnant) | All | 11.0 | 14.5 |

**Severity Threshold Matrix**
| Category | Severe (Grade 3) | Moderate (Grade 2) | Mild (Grade 1) |
| :--- | :--- | :--- | :--- |
| Children (0–4) | < 7.0 | < 10.0 | < 11.0 |
| Children (5–11) | < 8.0 | < 11.0 | < 11.5 |
| Children (12–14) | < 8.0 | < 11.0 | < 12.0 |
| Men (15+) | < 8.0 | < 11.0 | < 11.9 |
| Women (Non-Pregnant)| < 8.0 | < 11.0 | < 12.0 |
| Women (Pregnant) | < 7.0 | < 10.0 | < 11.0 |

#### 9c. Logic: Evaluate Anemia and Severity (Grade 0–3)
Compare the predicted Hemoglobin against the thresholds loaded from the tables above. **If the Hemoglobin drops below the "Mild" threshold, they are flagged as officially Anemic.** 
If they are Anemic, you check exactly how bad it is by seeing if it crosses the "Moderate" or "Severe" thresholds and assign Grade 1, 2, or 3. If they are not Anemic, they remain Grade 0 (Normal).

#### 9d. Logic: Calculate Anemia Risk Probability (%)
We calculate the distance between the Minimum Healthy Target and the Actual Hemoglobin using a mathematical "Sigmoid" curve formula (slope `k=2.5`).

#### 💡 The Complete Java Code for Step 9:

```java
// 1. Convert Normalized Hgb to Actual g/dL
float MIN_HGB = 7.0f;
float MAX_HGB = 17.4f;
float hgbValue = (predictedNormalizedHgb * (MAX_HGB - MIN_HGB)) + MIN_HGB;

// 2. Fetch Normal Ranges & Thresholds (Dynamically based on patient via above matrix)
// Example below represents a loaded "Women Non-Pregnant 15+" profile: 
float minNormalHb = 12.0f;
float maxNormalHb = 15.5f;
float severeThreshold = 8.0f;
float moderateThreshold = 11.0f;
float mildThreshold = 12.0f;

// 3. Determine Boolean Status Flags
boolean isAnemic = hgbValue < mildThreshold;
boolean isHigh = hgbValue > maxNormalHb;

// 4. Clinical Severity Logic (0, 1, 2, 3)
int severityGrade = 0; // Default: 0
String severityLabel = "Normal";

if (isAnemic) {
    if (hgbValue < severeThreshold) {
        severityGrade = 3; 
        severityLabel = "Severe Anemia";
    } else if (hgbValue < moderateThreshold) {
        severityGrade = 2; 
        severityLabel = "Moderate Anemia";
    } else {
        severityGrade = 1; 
        severityLabel = "Mild Anemia";
    }
} else if (isHigh) {
    severityGrade = 0; 
    severityLabel = "High Hemoglobin";
}

// 5. Calculate Anemia Risk Probability function
float dist = minNormalHb - hgbValue;
float k = 2.5f; 
double anemiaProbability = 1.0 / (1.0 + Math.exp(-k * dist));
float riskPercentage = (float) (anemiaProbability * 100.0);
```

---

### 📌 STEP 10: Display Results

Example Output mapping all properties:

```java
String displayResult = String.format(
    "Hemoglobin: %.1f g/dL (Normal: %.1f - %.1f)\n" +
    "Status: %s (Grade %d)\n" +
    "Risk Probability: %.1f%%\n" +
    "Is Anemic: %b\n" +
    "Is High: %b",
    hgbValue, minNormalHb, maxNormalHb,
    severityLabel, severityGrade,
    riskPercentage,
    isAnemic,
    isHigh
);
System.out.println(displayResult);
```

**Output Console:**
```
Hemoglobin: 10.5 g/dL (Normal: 12.0 - 15.0)
Status: Moderate Anemia (Grade 2)
Risk Probability: 97.7%
Is Anemic: true
Is High: false
```

---

## 📊 Workflow Diagram (Complete Pipeline)

```text
User provides Image, Age, Gender
         ↓
TensorFlow Lite Model Predicts Normalized Hemoglobin 
         ↓
App Post-Processing: Inverse Scale to calculate Actual g/dL
         ↓
App Post-Processing: Is hgb < normal threshold? 
         ↓
YES -> Flag isAnemic = true -> Assign Grade 1 (Mild), 2 (Moderate), or 3 (Severe)
NO  -> Flag isAnemic = false -> Assign Grade 0 (Normal or High)
         ↓
App Post-Processing: Calculate Risk % using Sigmoid(minHb - hgb)
         ↓
Display Dashboards & Alerts to User
```
