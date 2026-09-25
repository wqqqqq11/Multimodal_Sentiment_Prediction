# 问题三模型求解运行摘要

## 数据输入

读取问题三独立预处理目录中的 train、valid、test 与 attachment4_aligned；训练集只用于参数和原型学习，验证集用于早停与超参数选择，附件4只做最终推理。

## 参数初始化

文本骨干可由问题二同构参数热启动；选择器温度由高到低退火。Top-K、原型融合系数 alpha 和各损失权重必须在验证集调试，alpha 必须位于 [0,1]。

## 模型调用

先由上下文选择器选出全局 Top-K 证据，再把被选中的局部原始载荷交给受限预测器；联合类别/强度原型和充分性、完备性、稳定性反事实约束。预测器不接收完整选择概率向量。

## 结果输出

输出极性、强度、三模态 Shapley 作用程度、主要模态、逐证据删除重要性、训练原型匹配及原始片段定位。低置信时间映射和模态失效均显式标记。

## 本次产物

- valid: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\predictions\\valid_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\explanations\\valid_evidence.csv', 'metrics': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\metrics\\valid_metrics.json', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\reports\\valid_explanation_cards.md'}`
- test: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\predictions\\test_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\explanations\\test_evidence.csv', 'metrics': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\metrics\\test_metrics.json', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\reports\\test_explanation_cards.md'}`
- attachment4: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\predictions\\attachment4_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\explanations\\attachment4_evidence.csv', 'combined': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\predictions\\attachment4_prediction_and_explanation.csv', 'jsonl': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\explanations\\attachment4_explanations.jsonl', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\reports\\attachment4_explanation_cards.md'}`
- error_attribution: `D:\PythonWorkplace\Multimodal_Sentiment_Prediction\outputs\problem3\modeling\runs\20260924_231918_026\metrics\validation_error_attribution.json`
- figures: `['D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\training_history.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\validation_performance.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\validation_modality_contributions.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\validation_explanation_fidelity.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\attachment4_evidence_heatmap.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260924_231918_026\\figures\\validation_error_attribution.png']`
