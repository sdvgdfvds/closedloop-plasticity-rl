# 消融图表替换清单（可直接放论文）

1. 图5：`Fig5_ablation_components.png`
注释：组件消融结果。比较 PPO、PPO+Cycle、P3O-static、P3O-dynamic 在各环境的最终性能（Final Avg10 Reward），误差线为标准差。

2. 图6：`Fig6_alpha_static_vs_dynamic_curve.png`
注释：动态 α 与静态 α（α=0.4）在训练过程中的奖励曲线对比。横轴为环境交互步数（百万），纵轴为 Avg10 Reward。

3. 图7：`Fig7_alpha_gain_bar.png`
注释：动态 α 相对静态 α 的最终性能增益（Dynamic - Static）。柱体高于0表示动态 α 更优，低于0表示静态 α 更优。

4. 表5：`Table5_component_ablation.png`
注释：组件消融表。给出各方法最终性能均值±标准差及逐步增量（Cycle 相对 PPO、Static 相对 Cycle、Dynamic 相对 Static）。

5. 表6：`Table6_alpha_ablation.png`
注释：α 消融表。给出静态/动态 α 的成对比较、相对增益和胜率（win rate）。

6. 表7：`Table7_efficiency_ablation.png`
注释：训练效率消融。给出达到阈值所需步数、训练过程平均奖励、平均 SPS。阈值定义为该环境最佳最终性能的 80%。
