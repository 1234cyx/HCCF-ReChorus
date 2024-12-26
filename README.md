# 机器学习大作业

### 模型与框架介绍

+ [HCCF模型步骤](https://github.com/1234cyx/HCCF-ReChorus/blob/main/RecommendGNN/HCCF%E6%A8%A1%E5%9E%8B%E6%AD%A5%E9%AA%A4.md) 介绍了HCCF和Rechorus框架进行模型训练的步骤。
+ [Rechorus框架模型流程](https://github.com/1234cyx/HCCF-ReChorus/blob/main/RecommendGNN/Rechorus%E6%A1%86%E6%9E%B6%E6%A8%A1%E5%9E%8B%E6%B5%81%E7%A8%8B.md) 梳理了ReChorus框架的基本流程。



### 框架融合

+ 在 `src/models/general` 中增加 [HCCF](https://github.com/1234cyx/HCCF-ReChorus/blob/main/RecommendGNN/src/models/general/HCCF.py) 模型文件，同时，在 `src/helpers`中增加 [HCCFRunner](https://github.com/1234cyx/HCCF-ReChorus/blob/main/RecommendGNN/src/helpers/HCCFRunner.py) 文件，实现了HCCF融合进Rechorus。
+ 部分测评结果保存在 `log` 中
+ 模型权值保存在 `model`中



### 声明
**本仓库融合了Rechorus框架和HCCF模型，仅供学习使用，部分使用到的源代码地址如下所示：**

+ [Rechorus](https://github.com/THUwangcy/ReChorus)
+ [HCCF](https://github.com/akaxlh/HCCF)



