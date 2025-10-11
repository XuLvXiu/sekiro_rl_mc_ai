
## 动机与成效


## 演示视频

三周目白金的狼 再战 稀世强者弦一郎

https://www.bilibili.com/video//

## 游戏设置

- steam 开始游戏
- 游戏设置：图像设定 -- 屏幕模式设置为窗口，游戏分辨率调整为1280*720
- 游戏设置：按键设置 -- 重置视角/固定目标设置为 q，跳跃键为 f，垫步/识破键为空格，鼠标左键攻击，右键防御。
- 游戏设置：网络设定 -- 标题画面启动设定为 离线游玩.  目的是为了关闭残影，满地的残影太烦人了。
- 保持游戏窗口在最上层，不要最小化或者被其它的全屏窗口覆盖。

## 训练

- 确认环境

- 开始训练

## 预测

进入游戏，

在 cmd 窗口中运行：
```
python main.py 
```

等待模型加载完，

按 q 键锁定敌方

按 ] 键, 就会针对敌方的出招自动做出预测动作了。

再次按下 ] 键，会停止预测。

按 Backspace 键，退出程序。

有两种情况，AI 并不会处理：

其一，当游戏角色躺在地上的时候，AI 不会按垫步键快速起身，需要玩家来处理一下。

其二，喝血瓶也是一个很难的事情，包括一些人类玩家，都找不好喝血瓶的时机，AI也没有做这方面的专门训练，需要玩家按 ] 键暂停预测，找机会喝个血瓶，然后再次按下 ] 按恢复预测。

这主要是因为 AI 在训练的时候，只关注了敌兵的姿势，没有关注自己的 HP 与 姿势。

## 人工备份

模型的训练结果主要涉及到如下的几个文件：



如果要训练新模型的话，可能需要对老模型的这些数据进行备份。


## 大部分代码和思路来自以下仓库，感谢他们

- https://github.com/XR-stb/DQN_WUKONG
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5
- https://github.com/RongKaiWeskerMA/sekiro_play
- https://www.lapis.cafe/posts/ai-and-deep-learning/%E4%BD%BF%E7%94%A8resnet%E8%AE%AD%E7%BB%83%E4%B8%80%E4%B8%AA%E5%9B%BE%E7%89%87%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B
- https://blog.csdn.net/qq_36795658/article/details/100533639
