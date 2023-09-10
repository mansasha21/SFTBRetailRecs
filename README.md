# SFTBRetailRecs

Пайплайн для рекомендации товаров клиентам

# Установка
- `git clone https://github.com/mansasha21/SFTBRetailRecs.git`

# Запуск
```bash
python train.py && python eval.py
```

# Используемое решение

Пайплайн рекомендаций товаров:

* отбор товаров-кандидатов для предложения пользователю с использованием моделей коллаборативной фильтрации и матричных факторизаций,
* ранжирование отобранных кандидатов алгоритмами градиентного бустинга,
* выбор лучшего предложения исходя из бизнес-логики.

# Уникальность:

* Простая масштабируемость за счет использования докер контейнера,
* Учет двух сценариев возможного взаимодействия с пользователем:
  * предложение комплементарных товаров к товарам, находящимся в корзине,
  * предложение дополнительных товаров на основе паттернов, наблюдаемых у клиентов, и дополнительной информации о клиенте

# Стек используемых технологий:

`Python3`, `git`, `GitHub`, `Docker` - инструменты разработки  
`LightGBM`, `CatBoost`, `Implicit`, `Polars` - фреймворки машинного обучения    

# Сравнение моделей

Проводилось сравнение различных конфигураций моделей отбора товаров-кандидатов для предложения клиенту из семейства моделей коллаборативной фильтрации и матричной факторизации библиотеки Implicit в комбинации с моделью градиентного бустинга CatBoostRanker, выполняющей ранжирование отобранных кандидатов и выбор N лучших.

--В качестве устойчивого решения был выбран ансамбль из 5 моделей градиентного бустинга, с временем инференса 93.4 мс, так как ---он решает прогнозирует потенциальных клиентов с высоким (более 10% на отложенной выборке) результатом по предложенной метрике.


# Разработчики
| Имя                  | Роль           | Контакт               |
|----------------------|----------------|-----------------------|
| Суржиков Александр   | Data Scientist | https://t.me/mansasha |
| ---                  | ---            | ---                   |
| Назаренко Екатерина  | Data Scientist | https://t.me/cutttle  |
| ---                  | ---            | ---                   |
| Люльчак Павел        | Data Scientist | https://t.me/lllchak  |
| ---                  | ---            | ---                   |
