# Занятие 10.  Метрики качества ML‑моделей 

## Цель работы

Научиться рассчитывать в Python ключевые метрики качества ML-моделей, визуализировать их и делать обоснованные продуктовые выводы о готовности модели к запуску.

## Задание

Задание: для своего варианта данных рассчитайте ключевые метрики качества модели, постройте необходимые графики и кратко интерпретируйте результаты, указав сильные/слабые стороны модели.


## Варианты заданий

> [Архив с датасетами](../resources/metrics_datasets.zip)

| № варианта | Продуктовый кейс                                     | Тип задачи                 | Краткое описание кейса                                               |
| ---------- | ---------------------------------------------------- | -------------------------- | -------------------------------------------------------------------- |
| 1          | Отток клиентов подписочного сервиса                  | Классификация              | Модель оценивает, уйдёт ли клиент с платной подписки                |
| 2          | Мошенничество (фрод) в платежах                      | Классификация, дисбаланс   | Модель помечает подозрительные платёжные операции                   |
| 3          | Спам в чате службы поддержки                         | Классификация              | Модель отличает спам‑сообщения от нормальных обращений              |
| 4          | Клик по рекламному баннеру (CTR)                     | Классификация, редкий клик | Модель предсказывает, кликнет ли пользователь по баннеру            |
| 5          | Одобрение кредитной заявки                           | Классификация              | Модель решает, одобрить ли кредит потенциальному заёмщику           |
| 6          | Время доставки заказа                                | Регрессия                  | Модель прогнозирует время прибытия заказа к клиенту                 |
| 7          | Месячная выручка филиала                             | Регрессия                  | Модель прогнозирует выручку филиала на следующий месяц              |
| 8          | LTV (жизненная ценность) пользователя                | Регрессия                  | Модель оценивает суммарный доход от пользователя за период          |
| 9          | Рекомендации товаров в интернет‑магазине             | Рекомендательная система   | Модель формирует список товаров, которые стоит предложить           |
| 10         | Рекомендации плейлистов в музыкальном онлайн‑сервисе | Рекомендательная система   | Модель подбирает плейлисты, которые могут понравиться  пользователю |


## Подготовка среды

Установите/проверьте библиотеки:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Импорты, которые понадобятся почти в любом варианте датасета:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
```

## Часть А. Задачи классификации (варианты датасетов 1–5)

### Шаг 1. Загрузка и обзор данных

```python
df = pd.read_csv('variant_N_classification.csv')
print(df.shape)
print(df['y_true'].value_counts(normalize=True))
```

Обратите внимание на баланс классов — это критично для интерпретации accuracy (при сильном дисбалансе, как в кейсе фрода, метрика Accuracy может быть обманчиво высокой).

### Шаг 2. Расчёт метрик

```python
y_true, y_pred, y_score = df['y_true'], df['y_pred'], df['y_score']

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)
ll = log_loss(y_true, y_score)
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
print(f"ROC AUC={roc_auc:.3f}, PR AUC={pr_auc:.3f}, LogLoss={ll:.3f}")
print(cm)
```

### Шаг 3. Графики

**Confusion matrix:**
```python
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.title('Матрица ошибок')
plt.ylabel('Истинный класс'); plt.xlabel('Предсказанный класс')
plt.show()
```

**ROC-кривая:**
```python
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
plt.plot([0,1],[0,1],'--', color='grey', label='Случайная модель')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC-кривая'); plt.legend(); plt.show()
```

**PR-кривая:**
```python
precisions, recalls, _ = precision_recall_curve(y_true, y_score)
plt.figure(figsize=(5,4))
plt.plot(recalls, precisions, label=f'PR AUC={pr_auc:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall кривая'); plt.legend(); plt.show()
```

### Шаг 4. Интерпретация 

- **Accuracy** — доля верных предсказаний в целом; малоинформативна при дисбалансе классов (например, в детекции фрода, где положительный класс редкий).
- **Precision** — какая доля объектов, помеченных моделью как класс 1, действительно относится к классу 1; важна, когда цена ложноположительного срабатывания высока (например, лишняя блокировка легитимной транзакции).
- **Recall** — какая доля реальных объектов класса 1 была найдена моделью; критична, когда цена пропуска (ложноотрицательного) велика (например, пропущенный случай фрода или оттока).
- **F1-score** — баланс Precision и Recall, удобен при дисбалансе.
- **ROC AUC** — способность модели ранжировать позитивные объекты выше негативных независимо от порога; устойчива к дисбалансу лишь отчасти.
- **PR AUC** — более информативна, чем ROC AUC, при сильном дисбалансе классов.
- **Logloss** — штрафует модель за уверенные, но неверные предсказания; важен, если продукту нужны калиброванные вероятности (например, для скоринга риска).

Свяжите каждую метрику с конкретным продуктовым риском вашего кейса — не пишите определение метрики, а объясните, что она значит для бизнеса/пользователя в этом кейсе.

## Часть Б. Задачи регрессии (варианты 6–8)

### Расчёт метрик

```python
df = pd.read_csv('variant_N_regression.csv')
y_true, y_pred = df['y_true'], df['y_pred']

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, MAPE={mape:.3%}")
```

### Графики

```python
residuals = y_true - y_pred

fig, ax = plt.subplots(1,3, figsize=(15,4))
ax[0].scatter(y_true, y_pred, alpha=0.5)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax[0].plot(lims, lims, '--', color='red')
ax[0].set_xlabel('y_true'); ax[0].set_ylabel('y_pred'); ax[0].set_title('Факт vs Прогноз')

ax[1].hist(residuals, bins=30)
ax[1].set_title('Распределение остатков'); ax[1].set_xlabel('Остаток')

ax[2].scatter(y_pred, residuals, alpha=0.5)
ax[2].axhline(0, color='red', linestyle='--')
ax[2].set_xlabel('y_pred'); ax[2].set_ylabel('Остаток'); ax[2].set_title('Остатки vs Прогноз')
plt.tight_layout(); plt.show()
```

### Интерпретация

- **MAE** — средняя абсолютная ошибка в тех же единицах, что и целевая переменная; легко интерпретируется бизнесом (например, "модель ошибается в среднем на 6 минут").
- **RMSE** — сильнее штрафует крупные ошибки, чем MAE; полезна, если крупные промахи особенно дорого стоят продукту.
- **R²** — доля дисперсии целевой переменной, объяснённая моделью; показывает общее качество аппроксимации, но не абсолютную точность в единицах бизнеса.
- **MAPE** — ошибка в процентах, удобна для сравнения между сегментами, но нестабильна при значениях, близких к нулю.
- Форма графика остатков (нет ли систематического смещения или веерообразного разброса) говорит о наличии систематических ошибок модели, что важно указать в выводе.

## Часть В. Задачи рекомендаций (варианты 9–10)

### Парсинг данных

```python
df = pd.read_csv('variant_N_recsys.csv')
df['recommended_items'] = df['recommended_items'].apply(lambda s: [int(x) for x in s.split(';')])
df['relevant_items'] = df['relevant_items'].apply(lambda s: [int(x) for x in s.split(';')])
```

### Функции метрик

```python
def precision_at_k(rec, rel, k):
    rec_k = rec[:k]
    return len(set(rec_k) & set(rel)) / k

def recall_at_k(rec, rel, k):
    rec_k = rec[:k]
    return len(set(rec_k) & set(rel)) / len(rel) if rel else 0

def average_precision(rec, rel, k):
    hits, score = 0, 0.0
    for i, item in enumerate(rec[:k], start=1):
        if item in rel:
            hits += 1
            score += hits / i
    return score / min(len(rel), k) if rel else 0

def reciprocal_rank(rec, rel):
    for i, item in enumerate(rec, start=1):
        if item in rel:
            return 1 / i
    return 0

def ndcg_at_k(rec, rel, k):
    dcg = sum(1/np.log2(i+1) for i, item in enumerate(rec[:k], start=1) if item in rel)
    idcg = sum(1/np.log2(i+1) for i in range(1, min(len(rel), k)+1))
    return dcg / idcg if idcg > 0 else 0
```

### Расчёт по всем пользователям

```python
K = 10
df['precision_k'] = df.apply(lambda r: precision_at_k(r['recommended_items'], r['relevant_items'], K), axis=1)
df['recall_k'] = df.apply(lambda r: recall_at_k(r['recommended_items'], r['relevant_items'], K), axis=1)
df['ap'] = df.apply(lambda r: average_precision(r['recommended_items'], r['relevant_items'], K), axis=1)
df['rr'] = df.apply(lambda r: reciprocal_rank(r['recommended_items'], r['relevant_items']), axis=1)
df['ndcg'] = df.apply(lambda r: ndcg_at_k(r['recommended_items'], r['relevant_items'], K), axis=1)

MAP = df['ap'].mean()
MRR = df['rr'].mean()
mean_ndcg = df['ndcg'].mean()
print(f"Precision@{K}={df['precision_k'].mean():.3f}, Recall@{K}={df['recall_k'].mean():.3f}")
print(f"MAP={MAP:.3f}, MRR={MRR:.3f}, NDCG@{K}={mean_ndcg:.3f}")
```

### Графики

```python
ks = range(1, K+1)
prec_by_k = [df.apply(lambda r: precision_at_k(r['recommended_items'], r['relevant_items'], k), axis=1).mean() for k in ks]
rec_by_k = [df.apply(lambda r: recall_at_k(r['recommended_items'], r['relevant_items'], k), axis=1).mean() for k in ks]

plt.figure(figsize=(6,4))
plt.plot(ks, prec_by_k, marker='o', label='Precision@k')
plt.plot(ks, rec_by_k, marker='o', label='Recall@k')
plt.xlabel('k'); plt.ylabel('Значение метрики'); plt.title('Precision@k и Recall@k')
plt.legend(); plt.show()

plt.figure(figsize=(6,4))
plt.bar(['MAP','MRR', f'NDCG@{K}'], [MAP, MRR, mean_ndcg])
plt.title('Агрегированные метрики рекомендаций'); plt.ylabel('Значение')
plt.show()
```

### Интерпретация

- **Precision@k** — доля релевантных объектов среди топ-k рекомендаций; ключевая метрика для оценки "качества выдачи" на экране пользователя.
- **Recall@k** — какая доля всех релевантных объектов пользователя попала в топ-k; важна, если цель — покрыть максимум интересов пользователя.
- **MAP** — усреднённая по пользователям точность с учётом позиции релевантных объектов; хорошо отражает общее качество ранжирования.
- **MRR** — насколько быстро (на какой позиции) появляется первый релевантный объект; критична для сценариев "один клик — одна покупка/прослушивание".
- **NDCG@k** — учитывает и релевантность, и позицию объектов в выдаче; хорошо подходит, если релевантность не бинарна или порядок выдачи важен для UX.

## Общие правила оформления вывода

1. Формулируйте вывод не в терминах метрик ("recall = 0.82"), а в терминах продукта: что это значит для пользователя, бизнеса, рисков.
2. Явно укажите, какая метрика в вашем кейсе наиболее приоритетна и почему (например, для детекции фрода — Recall важнее Precision; для рекомендаций плейлистов — NDCG важнее MAP).
3. Завершите ноутбук однозначным заключением: модель пригодна к запуску / пригодна с оговорками / требует доработки, — с указанием, что именно нужно улучшить.

## Рекомендуемые руускоязычные названия метрик

- accuracy — доля верных предсказаний;
- precision — «точность по положительному классу;
- Recall — полнота по положительному классу;
- F1-score — сбалансированная оценка точности и полноты;
- ROC AUC — площадь под ROC-кривой;
- PR AUC — площадь под кривой «точность–полнота;
- logloss — логарифмическая функция потерь;
- MAE — средняя абсолютная ошибка;
- MSE — среднеквадратичная ошибка;
- RMSE — квадратный корень из среднеквадратичной ошибки;
- R² — коэффициент детерминации;
- MAPE — средняя абсолютная ошибка в процентах;
- precision@k — «точность в топ-k рекомендаций;
- recall@k — «полнота в топ-k рекомендаций;
- MAP — «средняя точность по пользователям;
- MRR — «средний обратный ранг первого релевантного объекта;
- NDCG — Normalized Discounted Cumulative Gain, нормированная дисконтированная суммарная полезность выдачи.

## Типичные ошибки, за которые снижается балл

- Расчёт Accuracy без учёта дисбаланса классов и без комментария об этом.
- Отсутствие подписей осей/легенды/заголовков на графиках.
- Интерпретация метрики как определения из учебника, без привязки к продуктовому кейсу.
- Отсутствие итогового вывода о пригодности модели.
- Использование только одного порога (0.5) без обсуждения, что ROC/PR-кривые показывают поведение при разных порогах.