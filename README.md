# Обучение модели  для классификации 3D объектов

## Описание

Репозиторий содержит код, необходимый дял обучения модели и подготовки данных для [репозитория с демо-приложением](https://github.com/KernelA/made-ml-demo-app)


## Зависимости

1. Python 3.7 или выше.
2. Anaconda
3. Видеокарта с поддержкой CUDA 10.2 или выше. Возможно использование CUDA 10.1 или CUDA 10.0, но файл с окружением на это не расчитан.

## Как запустить

Создать окружение:
```
conda env create -n env_name --file ./conda-env.yml
conda activate env_name
```

В проекте используется DVC. Основные этапы можно посмотреть через команду:
```
dvc dag
```

Для обучения модели достаточно выполнить команду:

```
dvc repro export_model
```
*Стоит заметить, что полная воспроизводимость не гарантируется.* Все файлы для модели хранятся в Git LFS.
