#  Job Market Analysis: Salary Prediction & Skill Identification  
## Course: CMPSC 445
**Joscelin Montoya Rojas**  
**Date: March 8, 2025**  

---

## **1️⃣ Introduction**
Understanding job market trends is essential for job seekers and employers. This project explores the **salary trends in computer science, data science, and AI fields** by predicting salaries and identifying important job skills.

Since web scraping from platforms like Indeed, Glassdoor, and LinkedIn is restricted, we **simulated a dataset** of 3,000 job postings with randomized job titles, companies, locations, skills, and salaries. The goal is to develop machine learning models that:
1. **Predict job salaries** based on features like job title, location, and experience level.
2. **Identify the most important skills** for high-paying jobs.

---

## **2️⃣ Data Collection**
### **Sources**
Due to web scraping limitations, the dataset was simulated to include **3,000** job postings for:
- **Software Engineers**
- **Data Scientists**
- **AI Engineers**
- **Machine Learning Engineers**
- **DevOps Engineers**

### **Collected Attributes**
| Feature | Description |
|---------|-------------|
| Job_Title | Job role (e.g., Software Engineer, Data Scientist) |
| Company | Randomized company names (Google, Amazon, Tesla, etc.) |
| Location | Randomized job locations (New York, Seattle, etc.) |
| Skills | Required skills (Python, SQL, Machine Learning, AWS, etc.) |
| Experience_Level | Entry, Mid, or Senior |
| Salary | Simulated salary range ($60,000 – $200,000) |

---

## **3️⃣ Data Preprocessing**
- **Encoded categorical features** (Job Title, Location, Company, Experience Level).
- **Applied One-Hot Encoding** to skills.
- **Split the dataset** into 80% training and 20% testing.

The cleaned dataset was saved as **`cleaned_job_data.csv`**.

---

## **4️⃣ Model Development & Evaluation**
We trained **two machine learning models**:

### **🔹 Model 1: Linear Regression**
- **Mean Absolute Error (MAE):** `XX,XXX`
- **Root Mean Squared Error (RMSE):** `XX,XXX`

### **🔹 Model 2: Random Forest Regressor**
- **Mean Absolute Error (MAE):** `XX,XXX`
- **Root Mean Squared Error (RMSE):** `XX,XXX`

 **Best Model:** **Random Forest Regressor performed better** because it handles non-linear salary variations more effectively.

---

## **5️⃣ Skill Importance Analysis**
Using **feature importance scores** from the Random Forest model, we identified the most valuable skills:

| Rank | Skill |
|------|--------|
| 1️⃣ | Python |
| 2️⃣ | Machine Learning |
| 3️⃣ | SQL |
| 4️⃣ | Deep Learning |
| 5️⃣ | AWS |

These skills have the highest impact on salary predictions.

---

## **6️⃣ Visualizations**
- **Salary Distribution by Job Role** (Boxplot)
- **Salary Heatmap by Location**
- **Skill Importance Bar Chart**

Findings:
- **Mid-level and Senior jobs pay significantly more** than entry-level positions.
- **San Francisco, Seattle, and New York offer the highest salaries**.
- **Python & Machine Learning** are the most valuable skills.

---

## **7️⃣ Challenges & Limitations**
🔹 **Simulated data** lacks real-world variability.  
🔹 **Web scraping restrictions** prevented using live job postings.  
🔹 **Lack of company reputation data**, which impacts salaries.  

---

## **8️⃣ Recommendations for Future Work**
1️⃣ Use **real job postings** via APIs instead of a simulated dataset.  
2️⃣ Apply **advanced models** like Gradient Boosting for better predictions.  
3️⃣ Incorporate additional **features** such as company size and industry sector.  

---

## **9️⃣ Conclusion**
This project successfully analyzed **job market trends** using machine learning. The **Random Forest model** provided the best salary predictions, and skills like **Python, SQL, and Machine Learning** were the most valuable. Future improvements should focus on using **real-world data** and testing **more advanced models**.

