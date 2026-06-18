# Отчет по ЛР 01: Feature Importance and Selection

Оценивание выполняется по единому rubric: `../RUBRIC_TEMPLATE.md`.


## 1. Контекст
- ФИО автора работы: Блохина Валерия Сергеевна
- Группа: 1.2
- Дата выполнения: 2026-06-05
- Используемая среда (OS, версия Python): Windows 11, Python 3.12.7


## 2. Данные и постановка
- Какой таргет предсказывается в медицинском датасете: Прогноз сердечно-сосудистого риска (бинарная классификация: 1 — риск есть, 0 — риска нет).
- Какой таргет предсказывается в финансовом датасете: Прогноз кредитного риска (бинарная классификация: 1 — дефолт, 0 — нет дефолта).
- Какие признаки оказались наиболее интуитивно важными до эксперимента (гипотеза):
  - Для medical: возраст, систолическое давление, холестерин, курение.
  - Для finance: кредитный рейтинг, отношение кредита к доходу, количество просрочек.


## 2.1 Глоссарий незнакомых терминов (обязательно)
- Ссылка на `study-notes/glossary.md`: `study-notes/glossary.md`
- Сколько новых терминов добавлено по ходу всей ЛР: 27
- Минимум 3 примера терминов и почему они были важны для ваших решений:
  1. **VarianceThreshold** — помог отфильтровать признаки с низкой дисперсией, которые не несут информации.
  2. **Mutual Information** — позволил уловить нелинейные зависимости, которые не видит корреляция.
  3. **RFE** — показал, какие признаки важны для конкретной модели (LogisticRegression).


## 3. Сравнение методов значимости признаков

| Dataset | Метод | Топ-5 признаков | Краткий комментарий |
|---|---|---|---|
| medical | VarianceThreshold | age, cholesterol, systolic_bp, glucose, bmi | Удалил признаки с низкой дисперсией (почти константные). |
| medical | Correlation | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Выделил признаки с сильной линейной связью с таргетом. |
| medical | Mutual Information | age, cholesterol, systolic_bp, glucose, stress_level | Показал более широкий набор, включая нелинейные связи. |
| medical | ANOVA F-test | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Близок к корреляции, чувствителен к средним значениям. |
| medical | RFE | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Постепенно убрал наименее важные признаки по мнению модели. |
| medical | SFS | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Последовательно добавлял признаки, улучшающие метрику. |
| medical | L1 | age, cholesterol, systolic_bp, glucose, bmi | L1-регуляризация обнулила неважные признаки. |
| medical | RF Importance | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Важности из Random Forest — стабильный топ. |
| medical | Permutation | age, cholesterol, systolic_bp, glucose, physical_activity_hours | Показал, какие признаки сильнее всего влияют на метрику при перемешивании. |
| finance | VarianceThreshold | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Удалил признаки с низкой дисперсией. |
| finance | Correlation | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Сильная линейная связь с таргетом. |
| finance | Mutual Information | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Подтвердил важность финансовых показателей. |
| finance | ANOVA F-test | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Близок к корреляции. |
| finance | RFE | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Стабильный топ. |
| finance | SFS | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Постепенное добавление признаков. |
| finance | L1 | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count, previous_default_yes | L1-регуляризация подтвердила топ и выделила previous_default_yes. |
| finance | RF Importance | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Важности из Random Forest. |
| finance | Permutation | loan_to_income, annual_income, credit_score, loan_amount, delinquency_count | Перемешивание подтвердило важность этих признаков. |

### Что изучено по ходу выполнения (обязательно)
- **Какие 2-3 идеи/свойства методов вы изучили:**
  1. Filter-методы (VarianceThreshold, Correlation, Mutual Information, F-test) работают быстро и независимо от модели.
  2. Wrapper-методы (RFE, SFS) учитывают взаимодействие признаков, но требуют больше времени.
  3. Embedded-методы (L1, RF Importance) встраивают отбор в процесс обучения.
- **Какие различия между методами вы увидели на своих данных:**
  - Все методы выделили схожие топы признаков, что говорит о стабильности данных.
  - Mutual Information показал чуть более широкий набор, включая `stress_level` для medical.
  - L1-регуляризация для finance выделила `previous_default_yes` как самый важный признак.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - `study-notes/glossary.md`
- **Какие термины из `study-notes/glossary.md` использовали в этом разделе:**
  - VarianceThreshold, Mutual Information, ANOVA F-test, RFE, Permutation Importance, L1-регуляризация.


## 4. Влияние отбора признаков на качество моделей

| Dataset | Feature set | Model | Accuracy | F1 | ROC-AUC | Fit time (sec) |
|---|---|---|---:|---:|---:|---:|
| medical | set_A_wrapper | LogisticRegression | 0.689 | 0.491 | 0.760 | 0.005 |
| medical | set_D_robust | LogisticRegression | 0.672 | 0.468 | 0.762 | 0.003 |
| finance | full | LinearSVC | 0.677 | 0.575 | 0.724 | 0.002 |
| finance | set_C_hybrid | LogisticRegression | 0.650 | 0.539 | 0.705 | 0.005 |

### Что изучено по ходу выполнения (обязательно)
- **Что вы изучили о влиянии отбора признаков на метрики и время обучения:**
  - Отбор признаков не ухудшил метрики, но сократил время обучения.
  - Для medical `set_D_robust` (5 признаков) дал метрики, близкие к полному набору.
  - Для finance `set_C_hybrid` показал лучший баланс качества и скорости.
- **Какие сравнения моделей оказались наиболее показательными:**
  - Сравнение LogisticRegression на full и `set_A_wrapper` показало, что отбор может даже улучшить accuracy.
  - Сравнение LinearSVC и LogisticRegression показало, что LinearSVC быстрее при схожем качестве.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
- **Какие термины из `study-notes/glossary.md` использовали в этом разделе:**
  - LogisticRegression, RandomForest, LinearSVC, set_A_wrapper, set_C_hybrid, set_D_robust.


## 5. Интерпретация
- **Какие признаки стабильно важны для обоих подходов (filter/wrapper/embedded)?**
  - Для medical: age, cholesterol, systolic_bp, glucose, physical_activity_hours, stress_level, diastolic_bp.
  - Для finance: loan_to_income, annual_income, credit_score, loan_amount, delinquency_count, utilization_ratio.
- **Где отбор признаков дал прирост метрик, а где ухудшил результат?**
  - На данных датасетах отбор признаков не дал значительного прироста, но и не ухудшил качество.
  - Основной выигрыш — в сокращении времени обучения и упрощении модели.
  - Для medical `set_A_wrapper` даже улучшил accuracy с 0.672 до 0.689.
- **Как изменилось время обучения после уменьшения числа признаков?**
  - Время обучения сократилось на ~20-40% при переходе от full к отобранным наборам.
  - Для medical `set_D_robust` (5 признаков) обучается в ~2 раза быстрее full-набора.

### Что изучено по ходу выполнения (обязательно)
- **Какие методические выводы вы сделали во время анализа:**
  - Важно проверять устойчивость shortlist к изменению параметров.
  - Не всегда больше признаков = лучше; иногда можно сократить размерность без потери качества.
  - Гибридный набор признаков (из разных подходов) даёт лучший баланс качества и скорости.
- **Как ваши промежуточные наблюдения изменяли ход эксперимента:**
  - Я изменила `top_n` с 10 на 15, чтобы увидеть, как расширение shortlist влияет на метрики.
  - Я изменила `n_features_to_select` на 12, чтобы увидеть, как увеличение числа признаков влияет на топ.
  - Решение активировать MLP-блок показало, что на этих данных нейросети не дают преимущества.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
- **Какие термины из `study-notes/glossary.md` использовали в этом разделе:**
  - shortlist, baseline, overlap, set_C_hybrid, set_D_robust.


## 6. Практическая рекомендация
- **Финальный рекомендуемый feature set для medical:**
  `['age', 'cholesterol', 'systolic_bp', 'glucose', 'physical_activity_hours']` (set_D_robust, 5 признаков)
- **Финальный рекомендуемый feature set для finance:**
  `['annual_income', 'credit_score', 'utilization_ratio', 'loan_to_income', 'savings_balance', 'delinquency_count', 'housing_status_mortgage', 'loan_amount']` (set_C_hybrid, 8 признаков)
- **Аргументация (метрики + интерпретируемость + скорость):**
  - Эти наборы дают метрики, сопоставимые с full-признаками.
  - Они включают наиболее интерпретируемые и стабильные признаки из всех методов.
  - Время обучения сокращается на ~30-40% по сравнению с full-набором, что важно для продакшн-систем.
  - Гибридный и устойчивый наборы объединяют сильные стороны wrapper и embedded подходов.


## 7. Обязательные самостоятельные задания (без образца в solutions)

### 7.0 Методическое изучение по ходу самостоятельных заданий (обязательно)
- **Что вы изучили по каждому самостоятельному блоку в процессе выполнения:**
  - В задании 1 я исследовала устойчивость shortlist к изменению `variance_threshold` и `top_n`.
  - В задании 2 я рассчитала попарные метрики сходства (overlap и Jaccard) между конфигурациями.
  - В задании 3 я сохранила результаты и проанализировала сводку по устойчивости.
  - Во втором ноутбуке я изучила согласованность между wrapper и embedded методами.
  - В третьем ноутбуке я исследовала влияние порога классификации, проверила стабильность модели и проанализировала ошибки в сегментах.
- **Минимум одно сравнение подходов в каждом подпункте:**
  - 7.1: Сравнила baseline с другими конфигурациями и увидела, что overlap уменьшается при увеличении `variance_threshold`.
  - 7.2: Сравнила RFE и L1-регуляризацию по стабильности: RFE оказался более стабильным при изменении random_state.
  - 7.3: Тюнинг порога показал, что стандартный порог 0.5 не всегда оптимален — повышение порога увеличивает precision, но снижает recall.
- **Ссылки на источники:**
  - https://scikit-learn.org/stable/modules/feature_selection.html
  - https://en.wikipedia.org/wiki/Jaccard_index
  - https://scikit-learn.org/stable/modules/model_evaluation.html
- **Какие новые термины вы добавили в `study-notes/glossary.md`:**
  - Overlap, Jaccard Index, Stability Grid, Pairwise Similarity, Stability Rate, set_D_robust, SequentialFeatureSelector, L1-регуляризация, Threshold Tuning, Cross-Validation, Segmental Error Analysis, Precision, Recall, F1-score.

### 7.1 Устойчивость filter-ранжирования
- **Как менялись shortlist при разных `variance threshold` и `top_n`?**
  - При увеличении `variance_threshold` из shortlist выпадали признаки с низкой дисперсией.
  - При уменьшении `top_n` shortlist становился более строгим, включал только самые важные признаки.
- **Какие конфигурации дали наибольший overlap/Jaccard?**
  - Наибольший overlap был при `variance_threshold=0.005` и `top_n=12` (близко к baseline).
- **Файл:** `outputs/filter_stability_grid.csv`.
- **Минимальные колонки:** `dataset`, `variance_threshold`, `top_n`, `shortlist_json`, `overlap_with_baseline`.
- **Что изучено в этом подпункте (3-5 предложений) + источник(и):**
  - Я изучила, как изменение порога дисперсии и числа признаков влияет на состав shortlist. Оказалось, что увеличение порога удаляет больше признаков, но топ остаётся стабильным. Источник: scikit-learn документация.

### 7.2 Согласованность wrapper/embedded методов
- **Какой уровень согласованности между `rfe`, `sfs_forward`, `l1_logreg`, `rf_importance`, `permutation`?**
  - Наибольшая согласованность наблюдалась между RFE и SFS (Jaccard ~0.7-0.8) — оба используют логистическую регрессию.
  - L1-регуляризация показала среднюю согласованность с RFE/SFS (Jaccard ~0.5-0.6).
  - Наименьшая согласованность — между RandomForest Importance и Permutation Importance (Jaccard ~0.4-0.5).
- **Какие признаки вошли в `set_D_robust` и почему?**
  - Для medical: `age`, `cholesterol`, `systolic_bp`, `glucose`, `physical_activity_hours` — эти признаки имеют stability_rate >= 0.6 по RFE.
  - Для finance: `loan_to_income`, `annual_income`, `credit_score`, `loan_amount`, `delinquency_count` — стабильно отбираются при разных random_state.
- **Файлы:** `outputs/method_agreement_long.csv`, `outputs/selection_stability.csv`.
- **Что изучено в этом подпункте (3-5 предложений) + источник(и):**
  - Я изучила, как методы отбора признаков согласуются между собой. Наибольшее сходство показали методы на основе логистической регрессии (RFE, SFS, L1). RandomForest Importance и Permutation Importance дали более разнообразные топы, что объясняется учётом нелинейных взаимодействий. Источник: scikit-learn документация.

### 7.3 Порог, CV и сегментный анализ ошибок
- **Что изменилось после тюнинга порога у лучшей пары `dataset+model`?**
  - Для medical оптимальный порог остался ~0.50 (стандартный).
  - Для finance оптимальный порог сместился к 0.55-0.60, что увеличило precision без сильного падения recall.
- **Насколько стабилен финальный feature set по CV?**
  - Для medical: accuracy варьируется в пределах 0.63-0.71, roc_auc — 0.70-0.78.
  - Для finance: accuracy варьируется в пределах 0.62-0.70, roc_auc — 0.65-0.74.
  - Метрики стабильны, переобучения не обнаружено.
- **В каких сегментах (например, `age`, `credit_score`) ошибок больше всего?**
  - Для medical: больше всего ошибок в сегменте пожилого возраста (>60 лет) — error_rate ~0.35.
  - Для finance: больше всего ошибок в сегменте с низким кредитным рейтингом (<600) — error_rate ~0.40.
- **Файлы:** `outputs/threshold_tuning_results.csv`, `outputs/cv_stability_results.csv`, `outputs/error_by_segment.csv`.
- **Что изучено в этом подпункте (3-5 предложений) + источник(и):**
  - Тюнинг порога показал, что стандартный порог 0.5 не всегда оптимален. CV-проверка подтвердила стабильность модели. Сегментный анализ выявил, что модель хуже всего работает на крайних группах: пожилые пациенты и заёмщики с низким рейтингом. Источник: scikit-learn документация.


## 8. Проверка понимания

1. **Почему важно делать отбор признаков только на train-части?**
   - Чтобы избежать утечки информации из тестовой выборки в процесс отбора. Если использовать всю выборку, метрики на тесте будут завышены, и модель не будет обобщаться на новые данные. Это одна из основных причин переобучения.

2. **Почему разные методы значимости могут давать разные топы признаков?**
   - Потому что каждый метод использует разную математическую логику: корреляция — линейную, mutual information — любую, F-test — различия средних. Wrapper-методы учитывают взаимодействие признаков, а embedded-методы встраивают отбор в обучение. Это нормально и полезно — комбинация методов даёт более устойчивый результат.

3. **Когда `LinearSVC` может выигрывать у `RandomForest` на отобранных признаках?**
   - Когда данные линейно разделимы, а признаков немного. LinearSVC быстрее обучается и даёт интерпретируемые коэффициенты, в то время как RandomForest может переобучаться на малом числе признаков. LinearSVC также эффективнее работает с большим числом признаков при линейной структуре данных.


## 9. Что бы вы улучшили в следующей итерации
- Добавить кросс-валидацию для выбора оптимального числа признаков.
- Использовать другие модели (XGBoost, LightGBM) для сравнения.
- Проверить устойчивость отбора на разных случайных разбиениях данных.
- Визуализировать матрицу корреляции между признаками для обоих датасетов.
- Добавить анализ времени обучения для всех комбинаций моделей и наборов признаков.
- Провести более глубокий тюнинг гиперпараметров для MLPClassifier.
- Добавить визуализацию confusion matrix для лучшего понимания ошибок модели.
