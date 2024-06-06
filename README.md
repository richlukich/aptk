# aptk

Для сегментации автомобилей были использованы две архитектуры [Mask2Former](https://github.com/facebookresearch/Mask2Former) и [OneFormer](https://github.com/SHI-Labs/OneFormer) предобученных на датасете **Cityscapes**.

Для установки **Mask2Former** и **OneFormer** использовалась [инструкция](https://github.com/SHI-Labs/OneFormer/blob/main/INSTALL.md).

Файлы:
-   **eval_json.py** - Создание json файлов с предсказаниями и разметкой
-   **eval_coco.py** - Подсчет метрик с помощью COCOtools
-   **ensemble.py**  - Предсказание с помощью ансамбля OneFormer и Mask2Former, который считался по формуле *y_pred = 0.7 * Mask2Former + 0.3 * OneFromer*
-   **calculate_miou.py** - Подсчет метрики mIoU

Результаты:

| Модель | mIoU |
|   :---:| :---:|
|OneFormer| 0.830|
|Mask2Former| **0.871**|
|Ensemble| 0.860|
