# 问题2运行说明

在项目根目录依次执行：

```powershell
python data_progressing/problem2_preprocess.py --overwrite
python data_progressing/problem2_validate.py
python scripts/problem2/solve_problem2.py --device auto
```

前两条命令分别生成、校验问题2对齐版预处理数据；第三条命令训练完整教师和缺失学生、完成验证/测试/附件3推理，并输出缺失规律、消融实验与图表。

首次运行建议先做端到端冒烟测试：

```powershell
python scripts/problem2/solve_problem2.py --device auto --smoke
```

每次运行写入独立的 `outputs/problem2/runs/<时间戳>/`，不会覆盖历史日志。正式运行还会更新 `outputs/problem2/submission/` 中的附件3预测、学生权重与复现配置。

并发由配置项 `data.num_workers` 控制；Windows 环境默认设为0最稳定，CPU和内存充足时可调整为2或4。显存不足时优先降低 `data.batch_size` 与 `data.eval_batch_size`。
