# Credit Risk Prediction & Explainable 

## Project Overview
โปรเจกต์นี้เป็นการวิเคราะห์ความเสี่ยงในการผิดนัดชำระหนี้ (Credit Risk / Loan Default) โดยใช้หลักการทางสถิติ (Hypothesis Testing) และสร้างโมเดล Machine Learning เพื่อทำนายความเสี่ยงของลูกค้า พร้อมประยุกต์ใช้ทฤษฎีเกม (SHAP Values) ในการอธิบายผลลัพธ์ของโมเดลให้ตอบโจทย์การตัดสินใจทางธุรกิจ

## Tech Stack & Tools
* **Data Processing:** Python, Pandas, NumPy
* **Statistical Analysis:** SciPy (Chi-Square, Welch's T-Test)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Explainable AI (XAI):** SHAP
* **Visualization:** Matplotlib, Seaborn

## 1. Data Cleansing & Outlier Detection
* ตรวจพบและจัดการ Logical Anomalies เช่น อายุผู้กู้ที่เกินจริง (144 ปี) และอายุงานที่เป็นไปไม่ได้ (123 ปี)
* เติมค่า Missing Values ในคอลัมน์อัตราดอกเบี้ยและอายุงานด้วยค่า Median เพื่อป้องกันผลกระทบจากความเบ้ของข้อมูล (Skewness)

## 2. Statistical Hypothesis Testing
เพื่อหลีกเลี่ยงอคติ (Bias) จากการดูข้อมูลด้วยตาเปล่า จึงได้ทำการทดสอบทางสถิติ:
* **Chi-Square Test (P-value < 0.001):** ยืนยันว่า "สถานะการครอบครองที่อยู่อาศัย" (เช่า vs มีบ้านของตัวเอง) มีความสัมพันธ์กับการผิดนัดชำระหนี้อย่างมีนัยสำคัญ
* **Independent T-Test (P-value < 0.001):** ยืนยันว่ากลุ่มลูกค้าที่เบี้ยวหนี้ ได้รับอัตราดอกเบี้ยเฉลี่ย (12.87%) สูงกว่ากลุ่มลูกค้าปกติ (10.49%) อย่างมีนัยสำคัญ

## 3. Machine Learning (Imbalanced Data Handling)
เนื่องจากชุดข้อมูลมีสัดส่วนคนจ่ายปกติมากกว่าคนเบี้ยวหนี้อย่างมาก จึงเลือกใช้พารามิเตอร์ `scale_pos_weight` ใน **XGBoost** เพื่อปรับสมดุลน้ำหนัก
* **Model:** XGBoost Classifier
* **Accuracy:** 91%
* **Recall (Class 1):** 78% (สามารถดักจับลูกค้าที่จะเบี้ยวหนี้ได้ถึง 78% จากทั้งหมด)
* **Precision (Class 1):** 80%

## 4. Model Interpretability with SHAP Values
เพื่อไม่ให้โมเดลเป็นเพียง Black Box ได้ใช้ SHAP Summary Plot เพื่อถอดรหัสปัจจัยความเสี่ยง พบว่า 3 ปัจจัยหลักที่ทำให้ลูกค้าเบี้ยวหนี้คือ:
1. **Loan Percent Income (สัดส่วนวงเงินกู้ต่อรายได้):** หากขอกู้เกินตัว จะเพิ่มความเสี่ยงสูงสุด
2. **Interest Rate (อัตราดอกเบี้ย):** ดอกเบี้ยที่แพงเป็นภาระหนักที่ผลักให้ลูกค้าทิ้งหนี้
3. **Person Income (รายได้ส่วนบุคคล):** ลูกค้ารายได้สูงมีแนวโน้มความเสี่ยงต่ำกว่าอย่างชัดเจน (Negative SHAP Value)

<img width="708" height="716" alt="Screenshot 2026-09-06 131331" src="https://github.com/user-attachments/assets/957ecd25-ddb5-4ba9-b7a1-768d3107b99f" />
<img width="411" height="306" alt="Screenshot 2026-09-06 131358" src="https://github.com/user-attachments/assets/c7900d9d-00e8-4aaf-a60e-639fdbbbec10" />
<img width="1500" height="716" alt="Screenshot 2026-09-06 131309" src="https://github.com/user-attachments/assets/f24224fa-6259-4d77-badb-f295a2917155" />

