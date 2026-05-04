# 数据完整性分析报告

## 汇总

- 可提取样本数: 1444
- 可提取患者数: 553
- 需人工核对问题数: 285
- 临床特征表患者数: 539
- 可提取患者中缺少临床特征: 16
- 临床特征表中缺少可提取样本: 2

## 可提取样本按年份

- 2024: 520
- 2025: 924

## 问题类型统计

- unparseable_image_name: 267
- missing_sequence_1_image: 9
- missing_temperature_csv: 6
- missing_front_image: 1
- missing_front_sequence_1: 1
- orphan_temperature_csv: 1

## 临床特征覆盖

- 可提取但不在临床表中的患者: AW006, BQ018, BV003, BY019, CA021, CE007, CJ007, CJ008, CN093, GCOO5, GJ005, GM104, GM106, GP105, QIU GUOFU邱国福, SONG SHUMEI 宋书梅
- 在临床表中但没有可提取样本的患者: BI017, BR014

## 提取规则

- 2024: 使用每个患者文件夹中的 1 号图像（支持 `ID1` 和 `ID-1`）及同名 CSV。
- 2025: 使用所有正面（`正`）JPG/JPEG/PNG 图像，以及 `热成像温度数据` 下同名 CSV。
- 2025 患者如果没有正面图像，或有正面但没有 1 号正面图像，先只写入日志，留待人工判定。
- 缺少图像/温度配对、无法解析患者编号的文件均写入问题明细 CSV。
