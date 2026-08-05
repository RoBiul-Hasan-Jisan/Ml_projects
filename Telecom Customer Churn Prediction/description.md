#  Telecom Customer Churn Prediction

##  Project Overview

Customer retention is one of the most critical challenges for telecommunication companies. This project focuses on building a machine learning-based customer churn prediction system that analyzes customer demographics, service usage patterns, account information, and billing behavior to identify customers who are likely to leave the service.

The primary objective is to understand the key factors influencing customer churn and develop a predictive model that enables telecom companies to take proactive retention strategies, improve customer satisfaction, and reduce revenue loss.

The project involves:
- Exploratory Data Analysis (EDA) to discover churn patterns and customer behavior
- Data preprocessing and feature engineering for machine learning
- Training and evaluating multiple classification models
- Identifying the most influential factors contributing to customer churn
- Selecting the best-performing model for churn prediction

---

#  Dataset Description

The dataset contains customer-level information, including demographic attributes, subscribed services, contract details, and billing information. The target variable is **Churn**, which indicates whether a customer has discontinued the telecom service.

##  Data Dictionary

| Feature | Description |
|---------|-------------|
| CustomerID | Unique identifier assigned to each customer |
| Gender | Customer gender |
| SeniorCitizen | Indicates whether the customer is a senior citizen (1 = Yes, 0 = No) |
| Partner | Indicates whether the customer has a partner |
| Dependents | Indicates whether the customer has dependents |
| Tenure | Number of months the customer has been with the company |
| PhoneService | Whether the customer uses phone services |
| MultipleLines | Whether the customer has multiple phone lines |
| InternetService | Type of internet service (DSL, Fiber optic, No service) |
| OnlineSecurity | Availability of online security service |
| OnlineBackup | Availability of online backup service |
| DeviceProtection | Availability of device protection service |
| TechSupport | Availability of technical support |
| StreamingTV | Subscription status of streaming TV service |
| StreamingMovies | Subscription status of streaming movie service |
| Contract | Customer contract type (Month-to-month, One year, Two years) |
| PaperlessBilling | Whether customer uses paperless billing |
| PaymentMethod | Customer payment method |
| MonthlyCharges | Monthly service charges |
| TotalCharges | Total accumulated charges |
| Churn | Target variable indicating whether the customer left the service |

---

#  Exploratory Data Analysis Insights

The analysis revealed several important patterns affecting customer churn:

###  Customer Demographics
- Customers without partners or dependents show a higher tendency to churn.
- Senior citizens demonstrate relatively lower churn compared to younger customers.

###  Customer Tenure
- Customer tenure has a strong inverse relationship with churn.
- New customers, especially those with less than 5 months of tenure, have a significantly higher probability of leaving the service.
- Long-term customers are more likely to remain loyal.

###  Contract Type
- Customers with month-to-month contracts have the highest churn rate.
- One-year and two-year contract customers show significantly better retention, suggesting that long-term contracts improve customer loyalty.

###  Billing Behavior
- Customers with high monthly charges and relatively low total charges are more likely to churn.
- Optimizing pricing strategies and offering personalized plans may help reduce customer loss.

###  Service Usage
- Customers using additional services such as streaming features generally show lower churn rates.
- Lack of services like online security, backup, and technical support is associated with higher churn probability.

---

#  Machine Learning Approach

Several classification algorithms were implemented and evaluated:

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN) Classifier

The models were evaluated using:

- Accuracy Score
- F1 Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

#  Model Performance & Results

Among all evaluated models, the **Random Forest Classifier** achieved the best performance:

 Accuracy: **82%**  
 Higher F1 Score compared to other models  
 Lower prediction error (MAE and MSE)  
 Better ability to capture complex customer behavior patterns  

The Random Forest model was selected as the final predictive model due to its robustness, strong generalization capability, and ability to identify important churn-driving features.

---

#  Key Churn Factors

Feature importance analysis identified the following variables as the strongest predictors of customer churn:

1. **Tenure**
2. **Contract Type**
3. **Monthly Charges**
4. **Total Charges**
5. **Internet Service Type**
6. **Customer Support Services**

These insights can help telecom companies design targeted customer retention strategies.

---

#  Business Impact

The developed churn prediction system can help telecom companies:

- Identify high-risk customers before they leave
- Provide personalized retention offers
- Improve customer satisfaction
- Reduce customer acquisition costs
- Increase long-term revenue and customer loyalty

---

#  Conclusion

This project demonstrates how machine learning can be effectively applied to customer behavior analysis and churn prediction. Through exploratory analysis and predictive modeling, important churn patterns were identified, and the Random Forest Classifier provided the most reliable prediction performance.

The final model can serve as a decision-support tool for telecom companies to proactively manage customer relationships and minimize churn.
