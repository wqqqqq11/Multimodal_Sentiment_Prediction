# 问题三模型求解运行摘要

## 数据输入

读取问题三独立预处理目录中的 train、valid、test 与 attachment4_aligned；训练集只用于参数和原型学习，验证集用于早停与超参数选择，附件4只做最终推理。

## 参数初始化

文本骨干由问题二同构参数热启动，完整问题二模型作为冻结教师；选择器温度与 Gumbel 噪声共同退火。Top-K 上限、局部窗口和各损失权重只在验证集调试。

## 模型调用

上下文选择器按样本分配自适应 Top-K，再把中心前后固定半径的局部载荷交给受限预测器；极性与强度共享有序约束。原型仅在训练后用于相似案例解释，不反馈预测。

## 结果输出

输出极性、强度、三模态 Shapley 作用程度、主要模态、逐证据删除重要性、训练原型匹配及原始片段定位。低置信时间映射和模态失效均显式标记。

## 本次产物

- valid: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\predictions\\valid_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\explanations\\valid_evidence.csv', 'metrics': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\metrics\\valid_metrics.json', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\reports\\valid_explanation_cards.md'}`
- test: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\predictions\\test_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\explanations\\test_evidence.csv', 'metrics': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\metrics\\test_metrics.json', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\reports\\test_explanation_cards.md'}`
- attachment4: `{'predictions': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\predictions\\attachment4_predictions.csv', 'evidence': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\explanations\\attachment4_evidence.csv', 'combined': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\predictions\\attachment4_prediction_and_explanation.csv', 'jsonl': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\explanations\\attachment4_explanations.jsonl', 'explanation_cards': 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\reports\\attachment4_explanation_cards.md'}`
- error_attribution: `D:\PythonWorkplace\Multimodal_Sentiment_Prediction\outputs\problem3\modeling\runs\20260925_094632_254\metrics\validation_error_attribution.json`
- figures: `['D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\training_history.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\validation_performance.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\validation_modality_contributions.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\validation_explanation_fidelity.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\attachment4_evidence_heatmap.png', 'D:\\PythonWorkplace\\Multimodal_Sentiment_Prediction\\outputs\\problem3\\modeling\\runs\\20260925_094632_254\\figures\\validation_error_attribution.png']`
