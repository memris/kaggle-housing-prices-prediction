# kaggle-housing-prices-prediction
Прогнозирование стоимости жилья (Kaggle's House Prices Competition)
# Описание проекта
Цель данного проекта — разработка модели машинного обучения для точного предсказания рыночной стоимости жилья. Проект выполнен в рамках соревнования Kaggle "House Prices: Advanced Regression Techniques".

Основная задача — построить модель регрессии, которая на основе различных характеристик дома (таких как площадь, количество комнат, район, год постройки и т.д.) сможет предсказать его итоговую цену продажи.

# Набор данных
Проект использует официальный набор данных соревнования — Ames Housing dataset. Этот датасет широко известен и используется в сообществе Data Science для практики в решении задач регрессии и продвинутого feature engineering.

train.csv - Обучающая выборка, содержащая 1460 записей с 79 признаками и целевой переменной SalePrice.
test.csv - Тестовая выборка, содержащая 1459 записей с 79 признаками, для которых необходимо предсказать SalePrice.
# Стек технологий
Python, Pandas & NumPy, Matplotlib & Seaborn, Scikit-learn, LightGBM.
# Основные этапы проекта
1. Разведочный анализ данных (EDA): Было проанализировано распределение целевой переменной SalePrice, выявлена и скорректирована сильная асимметрия с помощью логарифмического преобразования. С помощью тепловой карты корреляций и диаграмм рассеяния были изучены взаимосвязи между SalePrice и ключевыми признаками, такими как GrLivArea и OverallQual. На основе анализа были выявлены и удалены 2 аномальных выброса, которые могли негативно повлиять на обучение.
2. Feature Engineering и Preprocessing: Проведена комплексная работа по заполнению пропущенных значений, используя различные стратегии (заполнение нулями, модой, медианой). Был создан новый признак TotalSF (общая площадь дома) для повышения предсказательной силы. Все категориальные признаки были преобразованы в числовой формат с помощью One-Hot Encoding.
3. Моделирование и Оценка: Качество моделей оценивалось с помощью 5-фолдовой кросс-валидации и метрики RMSE. В качестве baseline была использована Линейная регрессия. Основной моделью был выбран градиентный бустинг LightGBM, который показал значительно лучший результат. Финальная модель LightGBM была обучена с предварительно подобранными гиперпараметрами, а итоговые предсказания были преобразованы обратно из логарифмической шкалы для получения финального результата.

# Итоговый вывод
В результате проделанной работы была построена и настроена модель градиентного бустинга, которая показала высокую точность предсказаний.

Итоговая модель LightGBM достигла метрики RMSLE = 0.1291 на приватном лидерборде соревнования, что является конкурентным и сильным результатом.

Файл с предсказаниями (submission.csv), сгенерированный ноутбуком, демонстрирует завершенный цикл проекта от анализа до получения конечного результата.

# Как запустить проект
### Данные
Для запуска проекта скачайте файлы `train.csv` и `test.csv` с [официальной страницы соревнования Kaggle](https://www.kaggle.com/competitions/home-data-for-ml-course/data) и поместите их в корневую директорию проекта.

```bash
# Клонируйте репозиторий
git clone https://github.com/YourUsername/kaggle-housing-prices-prediction.git

# Создайте и активируйте виртуальную среду
python3 -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```
---

# Kaggle Housing Prices Prediction
(English Version)

# Project Description
The goal of this project is to develop a machine learning model to accurately predict the market value of residential property. The project was created as a solution for the Kaggle competition: **"House Prices: Advanced Regression Techniques"**.

The main task is to build a regression model that can predict the final sale price based on various home characteristics (such as area, number of rooms, neighborhood, year built, etc.).

# Dataset
The project uses the official competition dataset — the **Ames Housing dataset**. This dataset is widely known in the Data Science community for practicing regression tasks and advanced feature engineering.

- **train.csv** - The training set containing 1460 records with 79 features and the target variable `SalePrice`.
- **test.csv** - The test set containing 1459 records with 79 features for which the `SalePrice` needs to be predicted.

# Tech Stack
Python, Pandas & NumPy, Matplotlib & Seaborn, Scikit-learn, LightGBM.

# Key Stages
1. **Exploratory Data Analysis (EDA):** Analyzed the distribution of the target variable `SalePrice`. Identified strong skewness and corrected it using logarithmic transformation. Investigated relationships between `SalePrice` and key features (such as `GrLivArea` and `OverallQual`) using correlation heatmaps and scatter plots. Based on this analysis, 2 anomalies (outliers) that could negatively affect training were identified and removed.
2. **Feature Engineering & Preprocessing:** Performed comprehensive missing value imputation using various strategies (zeros, mode, median). Engineered a new feature, `TotalSF` (Total Square Footage), to enhance predictive power. All categorical features were converted to numerical format using One-Hot Encoding.
3. **Modeling & Evaluation:** Model performance was evaluated using 5-fold cross-validation and the RMSE metric. Linear Regression was used as a baseline. **LightGBM** (Gradient Boosting) was selected as the primary model due to its superior performance. The final LightGBM model was trained with tuned hyperparameters, and the resulting predictions were transformed back from the logarithmic scale to obtain the final output.

# Results
As a result of this work, a gradient boosting model was built and tuned, demonstrating high prediction accuracy.

The final LightGBM model achieved a score of **RMSLE = 0.1291** on the private leaderboard, which is a competitive and solid result.

The `submission.csv` file generated by the notebook demonstrates the complete project cycle from analysis to final inference.

# How to Run
### Data
To run the project, download `train.csv` and `test.csv` from the [official Kaggle competition page](https://www.kaggle.com/competitions/home-data-for-ml-course/data) and place them in the root directory.

```bash
# Clone the repository
git clone https://github.com/YourUsername/kaggle-housing-prices-prediction.git

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```