Исследовательский Python-проект для ВКР по сравнению CNN и Vision Transformer на датасете Office-Home. Проект проверяет классификацию изображений при domain shift, semantic OOD, calibration и common corruptions.

Основные модели берутся из `timm`: DINOv3 ConvNeXt и DINOv3 ViT сравниваются на уровнях `Small` и `Base`; `Large` используется как scaling check; `ResNet-50` оставлен как исторический baseline.

## Содержание

- [Структура проекта](#структура-проекта)
- [Подготовка](#подготовка)
- [Данные](#данные)
- [Конфигурации](#конфигурации)
- [Запуск](#запуск)
- [Outputs](#outputs)

## Структура проекта

```text
vkr-cnn-vit/
├── configs/        # YAML-конфиги экспериментов
├── data/           # Office-Home dataset, не хранится в git
├── notebooks/      # анализ результатов и графики
├── outputs/        # checkpoints, logits, metrics, tables, не хранится в git
├── scripts/        # CLI-запуски обучения, оценки и агрегации
├── src/            # переиспользуемая логика проекта
├── README.md
├── pyproject.toml
└── uv.lock
```

Основные модули:

- `src/data.py` — чтение Office-Home, нормализация доменов, split-протоколы.
- `src/models.py` — `timm`-модели, feature pooling, transforms, режимы заморозки.
- `src/train.py` — orchestration обучения.
- `src/train_linear.py` — cached linear probing.
- `src/train_optim.py` — AdamW, warmup, cosine scheduler, LLRD.
- `src/train_loops.py` — train loop с gradient accumulation.
- `src/eval.py` — inference и сбор logits/labels/predictions/paths.
- `src/metrics.py` — classification, calibration, domain shift и OOD-метрики.
- `src/corruptions.py` — ImageNet-C-style corruptions через `imagecorruptions`.
- `src/ood.py` — semantic OOD report по MSP и Energy.

Основные Python-модули запуска:

- `scripts/train_lodo.py` — единый CLI для `lodo`, `cross_domain`, `in_domain`, `semantic_ood`.
- `scripts/make_semantic_ood_split.py` — фиксированный split known/unknown классов.
- `scripts/eval_ood.py` — semantic OOD evaluation.
- `scripts/eval_corruptions.py` — полный или subset corruption evaluation.
- `scripts/precompute_corruptions_s3s5.py` — precompute JPEG-shards и linear embeddings для выбранных severity.
- `scripts/eval_corruptions_precomputed.py` — corruption evaluation по precomputed JPEG-shards/embeddings.
- `scripts/aggregate.py` — сбор метрик в итоговые CSV-таблицы.

## Подготовка

Проект рассчитан на Python 3.10+ и `uv`.

Создать окружение и установить зависимости:

```bash
uv sync
```

Проверить lock-файл:

```bash
uv lock --check
```

Все команды запускаются из корня проекта.

## Данные

Основной датасет: Office-Home.

Ожидаемая структура:

```text
data/OfficeHomeDataset_10072016/
├── Art/
├── Clipart/
├── Product/
└── Real World/
```

Домены `Real World`, `Real_World` и `RealWorld` нормализуются к `RealWorld`.

Поддерживаемые протоколы:

- `lodo` — один домен полностью held-out, остальные домены используются для `train/val/id_test`.
- `cross_domain` — `Product + RealWorld -> Art + Clipart`.
- `in_domain` — `train/val/id_test` внутри одного домена.
- `semantic_ood` — known-классы используются для обучения, unknown-классы появляются только в `unknown_test`.

Во всех обучающих протоколах split стратифицированный по классу:

```text
train / val / id_test = 60 / 20 / 20
```

В `lodo` held-out домен не попадает ни в train, ни в validation. В `semantic_ood` unknown-классы не попадают в train/val/id_test.

## Конфигурации

Основные YAML-конфиги:

```text
configs/officehome_resnet50.yaml
configs/officehome_dinov3_small.yaml
configs/officehome_dinov3_base.yaml
configs/officehome_dinov3_large.yaml
```

Модели:


| Уровень       | CNN                     | ViT                 |
| ------------- | ----------------------- | ------------------- |
| Small         | `dinov3_convnext_tiny`  | `dinov3_vit_s_plus` |
| Base          | `dinov3_convnext_base`  | `dinov3_vit_b`      |
| Large scaling | `dinov3_convnext_large` | `dinov3_vit_l`      |


Baseline:

```text
resnet50
```

Режимы обучения:

- `linear_probe` — backbone frozen, обучается classifier head.
- `partial_finetune` — обучаются последние блоки backbone и classifier head.
- `full_finetune` — обучается вся модель; используется как дорогой upper-bound.

Общий recipe:

- optimizer: AdamW;
- scheduler: cosine annealing;
- epochs: `50`;
- early stopping: validation Macro-F1;
- input resolution: `256x256` для всех моделей;
- label smoothing, RandAugment, Mixup и CutMix выключены;
- AMP включён;
- gradient accumulation используется, если effective batch больше physical batch;
- warmup и LLRD применяются для `partial_finetune` и `full_finetune`.
- learning rate: `3e-4` для ResNet-50 и DINOv3 Small, `1e-4` для DINOv3 Base/Large;
- weight decay: `0.05`;
- patience: `7`.

Batch recipe:


| Уровень модели | Режим              | Physical batch | Effective batch |
| -------------- | ------------------ | -------------: | --------------: |
| ResNet/Small/Base | `linear_probe`     | 128 | 128 |
| ResNet/Small/Base | `partial_finetune` | 64  | 64  |
| ResNet/Small/Base | `full_finetune`    | 32  | 32  |
| Large             | `linear_probe`     | 128 | 128 |
| Large             | `partial_finetune` | 32  | 64  |
| Large             | `full_finetune`    | 8   | 32  |


Для `linear_probe` используется кэширование embeddings:

1. frozen backbone считает признаки для train/val/test;
2. embeddings сохраняются в `outputs/predictions/<run>/embeddings_*.npz`;
3. обучается только линейная голова;
4. `best.pt` сохраняется как полный checkpoint.

Для DINOv3 ViT во всех режимах используется `cls_patch_mean`: признаки строятся как `CLS token + mean patch tokens`. Register tokens исключаются из среднего. Для ConvNeXt и ResNet используется стандартный `timm` pooling.

Semantic OOD split зафиксирован в:

```text
configs/semantic_ood_split_seed42.json
configs/semantic_ood_split_seed42.yaml
```

Unknown classes:

```text
Folder, Laptop, Mop, Paper_Clip, Pencil,
Ruler, Sink, Speaker, ToothBrush, Trash_Can
```

## Запуск

### Один эксперимент

```bash
uv run python scripts/train_lodo.py \
  --config configs/officehome_dinov3_base.yaml \
  --model dinov3_vit_b \
  --mode linear_probe \
  --protocol lodo \
  --heldout-domain Art \
  --seed 42
```

In-domain:

```bash
uv run python scripts/train_lodo.py \
  --config configs/officehome_dinov3_large.yaml \
  --model dinov3_vit_l \
  --mode linear_probe \
  --protocol in_domain \
  --domain Product \
  --seed 42
```

Cross-domain:

```bash
uv run python scripts/train_lodo.py \
  --config configs/officehome_dinov3_large.yaml \
  --model dinov3_vit_l \
  --mode linear_probe \
  --protocol cross_domain \
  --seed 42
```

Semantic OOD:

```bash
uv run python scripts/train_lodo.py \
  --config configs/officehome_dinov3_large.yaml \
  --model dinov3_vit_l \
  --mode linear_probe \
  --protocol semantic_ood \
  --seed 42
```

### Экспериментальная матрица

Итоговые результаты получены для `202` checkpoint: `lodo`, `in_domain`, `cross_domain`, `semantic_ood`, три режима обучения и модели `ResNet-50`, DINOv3 ConvNeXt/ViT уровней `Small`, `Base`, `Large`.

Все отдельные обучения запускаются одним CLI `scripts/train_lodo.py`. Для повторения другого seed достаточно передать `--seed`, например:

```bash
uv run python scripts/train_lodo.py \
  --config configs/officehome_dinov3_base.yaml \
  --model dinov3_vit_b \
  --mode partial_finetune \
  --protocol lodo \
  --heldout-domain Clipart \
  --seed 123
```

Для `Large` scaling используются те же протоколы и режимы, но конфиг `configs/officehome_dinov3_large.yaml` и модели:

```text
dinov3_convnext_large
dinov3_vit_l
```

### Semantic OOD Evaluation

```bash
uv run python scripts/eval_ood.py \
  --checkpoint outputs/checkpoints/<run>/best.pt \
  --model dinov3_vit_b \
  --out outputs/metrics/<run>/ood.json
```

Сохраняются:

```text
outputs/predictions/<run>/ood_id_test.npz
outputs/predictions/<run>/ood_unknown_test.npz
outputs/metrics/<run>/ood.json
```

### Corruptions Evaluation

Полный ACS для одного checkpoint:

```bash
uv run python scripts/eval_corruptions.py \
  --checkpoint outputs/checkpoints/<run>/best.pt \
  --model dinov3_vit_l \
  --split target_test
```

По умолчанию считается:

```text
15 corruption types × 5 severity levels = 75 inference passes
```

Для быстрой проверки можно ограничить список corruption types и severity. Например, subset:

```text
16 checkpoints × 5 corruption types × severities 1/3/5 = 240 inference passes
```

Subset corruptions:

```text
gaussian_noise, motion_blur, fog, contrast, jpeg_compression
```

Для ручного subset-запуска:

```bash
uv run python scripts/eval_corruptions.py \
  --checkpoint outputs/checkpoints/<run>/best.pt \
  --model dinov3_vit_l \
  --split target_test \
  --corruptions gaussian_noise motion_blur fog contrast jpeg_compression \
  --severities 1 3 5 \
  --summary-name subset_ACS \
  --out outputs/metrics/<run>/corruptions_subset.csv
```

Для in-domain checkpoint использовать:

```bash
--split id_test
```

Для массовой оценки используется precompute JPEG-shards и embeddings для `linear_probe`. Полный ACS считается по всем `15 × 5 = 75` настройкам: `severity 1,2,3,4,5`.

```bash
uv run python scripts/precompute_corruptions_s3s5.py \
  --out outputs/corruption_cache/imagenet_c_full \
  --severities 1 2 3 4 5 \
  --image-size 256 \
  --embeddings
```

Оценка одного checkpoint по precomputed corruptions:

```bash
uv run python scripts/eval_corruptions_precomputed.py \
  --checkpoint outputs/checkpoints/<run>/best.pt \
  --cache outputs/corruption_cache/imagenet_c_full \
  --severities 1 2 3 4 5 \
  --out outputs/metrics/<run>/corruptions.csv
```

В текущем полном прогоне severity были посчитаны двумя частями и сохраняются как:

```text
outputs/metrics/<run>/corruptions_s3_s5.csv
outputs/metrics/<run>/corruptions_s1_s2_s4.csv
```

Notebook объединяет эти части в полный `ACS` и сохраняет компактную таблицу `outputs/tables/acs_full.csv`.

### Агрегация

```bash
uv run python scripts/aggregate.py
```

Агрегатор собирает:

```text
outputs/metrics/**/metrics.json
outputs/metrics/**/ood.json
outputs/metrics/**/corruptions.csv
outputs/metrics/**/corruptions_subset.csv
outputs/metrics/**/corruptions_s3_s5.csv
outputs/metrics/**/corruptions_s1_s2_s4.csv
```

## Outputs

Структура результатов:

```text
outputs/
├── checkpoints/
│   └── <run>/best.pt
├── logs/
│   └── <run>/
│       ├── config_used.yaml
│       └── history.csv
├── predictions/
│   └── <run>/
│       ├── embeddings_*.npz
│       ├── id_test.npz
│       ├── target_test.npz
│       ├── unknown_test.npz
│       ├── ood_id_test.npz
│       └── ood_unknown_test.npz
├── metrics/
│   └── <run>/
│       ├── metrics.json
│       ├── ood.json
│       ├── corruptions.csv
│       ├── corruptions_subset.csv
│       ├── corruptions_s3_s5.csv
│       └── corruptions_s1_s2_s4.csv
├── tables/
│   ├── acs_full.csv
│   ├── metrics_long.csv
│   ├── summary.csv
│   └── lodo_summary.csv
└── figures/
```

`metrics_long.csv` содержит long-format таблицу:

```text
protocol, model, mode, domain_or_heldout, metric, seed, value, source, run
```

`summary.csv` группирует результаты по:

```text
protocol, model, mode, domain_or_heldout, metric
```

и сохраняет:

```text
n, mean, std, mean_std
```

`lodo_summary.csv` собирает LODO по четырём held-out доменам:

```text
model, mode, seed, Art, Clipart, Product, RealWorld, mean_lodo_f1, worst_lodo_f1
```

`acs_full.csv` содержит компактную таблицу полного corruption score:

```text
protocol, model, mode, domain, seed, clean_macro_f1, acs_macro_f1, relative_drop, run
```

Если для одной пары `protocol/model/mode/domain_or_heldout/seed/metric/source` есть несколько запусков, агрегатор берёт последний run по timestamp. Старые `outputs` можно не удалять, если нужен свежий recipe.

Notebook `notebooks/results.ipynb` используется только для анализа таблиц, графиков и подготовки визуализаций для ВКР.
