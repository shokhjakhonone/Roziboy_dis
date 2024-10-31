# Система анализа поведения студентов

**"Основано на диссертации: "Анализ поведения студентов в электронных образовательных системах"**

     Автор диссертации: Рузибоев Самандар Кудрат Угли, ЭМС-271  
     Университет: Российский университет транспорта (РУТ) – МИИТ  
     Институт: Институт экономики и финансов (ИЭФ)  
     Направление: Прикладная информатика  
     Разработчик: Тухтамирзаев Шохжахон

## Описание проекта

Этот проект представляет собой веб-приложение для кластерного анализа данных о студентах, созданное на основе методов, описанных в диссертации. Приложение позволяет преподавателям и администраторам образовательных учреждений загружать данные о студентах, настраивать параметры кластеризации и анализировать поведение студентов в электронной образовательной среде.

Основные задачи проекта:
- Анализировать учебные данные студентов для выделения групп с похожим поведением.
- Обеспечить преподавателей полезной информацией для выявления потенциальных проблем у студентов и улучшения качества дистанционного обучения.

## Функции проекта

    - Загрузка данных студентов: Поддерживается загрузка данных в формате CSV (например, `student-mat.csv` из набора данных успеваемости студентов UCI).
    - Настройка методов кластеризации: Пользователь может выбрать алгоритм K-means или DBSCAN для анализа данных и гибко настроить параметры для оптимальной кластеризации.
    - Автоматическое создание графика: Приложение визуализирует результаты кластеризации, что упрощает интерпретацию полученных данных.
    - Логирование операций: Лог-файл сохраняет ключевые операции, помогая отслеживать ход выполнения и диагностировать возможные ошибки.

## Установка

1. **Клонируйте проект и перейдите в директорию проекта**:
   
       git clone https://github.com/shokhjakhonone/Roziboy_dis.git
       cd Roziboy_dis
   

2. **Установите необходимые зависимости**:
   
       pip install flask pandas scikit-learn matplotlib seaborn
   

3. **Запуск приложения**:
   
       python app.py
   

4. **Откройте приложение в браузере**:
   Перейдите по адресу [http://127.0.0.1:5000](http://127.0.0.1:5000) для работы с веб-интерфейсом.

## Использование

1. **Загрузите файл CSV**: На главной странице выберите файл с данными студентов (например, `student-mat.csv`).
2. **Выберите метод кластеризации**:
   - **K-means**: Требуется задать количество кластеров.
   - **DBSCAN**: Требуется задать параметры `EPS` и `Минимальное количество выборок`.
3. **Нажмите "Анализировать"**: Приложение выполнит кластеризацию, сохранит результаты в файл `student_clusters.csv` и создаст график.
4. **Просмотрите результаты**: На странице результатов отобразятся кластерные данные и визуализация кластеров, что поможет понять поведение и успеваемость студентов.

## Методология

Данный проект реализует два метода кластерного анализа:
- **K-means**: Алгоритм для разбиения данных на фиксированное число кластеров. Позволяет разделить студентов на группы по схожим характеристикам, таким как посещаемость и оценки.
- **DBSCAN**: Алгоритм плотностного анализа, выделяющий кластеры с плотной концентрацией точек. Удобен для выявления аномалий и групп с нестандартными учебными показателями.

Оба метода помогают преподавателям и администраторам понять поведение студентов и потенциально выявить группы, нуждающиеся в дополнительной поддержке.

## Логирование

Каждое действие сохраняется в файл `app.log`, включая:
- Загрузку данных и выполнение кластеризации
- Сохранение и визуализацию результатов

Это помогает отслеживать работу приложения и устранять потенциальные проблемы.

## Дальнейшие улучшения

Проект можно расширить, добавив:
- Дополнительные методы анализа данных.
- Интерактивные элементы и фильтры для углубленного анализа.
- Возможность анализа прогрессии успеваемости студентов по времени.

## Заключение

Проект демонстрирует применение методов кластерного анализа, описанных в диссертации, и предоставляет преподавателям инструмент для улучшения качества дистанционного обучения за счет анализа поведения студентов.
