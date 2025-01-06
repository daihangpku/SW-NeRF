# NeRF_Fans
## Quick Start
create a virtual environment and clone the repository
  ```bash
  conda create -n nerf_fans python=3.9
  conda activate nerf_fans
  git clone <repository_url>
  cd NeRF_Fans
  ```
get torch ready(replace XXX with your cuda version)
```bash  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
```
## Issues

### 系统建议
- 推荐整一个双系统，推荐 Ubuntu 22.04。[安装教程](https://www.bilibili.com/video/BV1Cc41127B9/?spm_id_from=333.337.search-card.all.click&vd_source=27087b1cb0b05427c1fcb74760d4296f)   
-  WSL 也能玩。
### 工具推荐
- [论文管理软件zotero](https://www.zotero.org/download/)
- git教程
  - [Git相关文字教程](https://csdiy.wiki/%E5%BF%85%E5%AD%A6%E5%B7%A5%E5%85%B7/Git/)
  - [Git可视化学习（推荐）](https://learngitbranching.js.org/?locale=zh_CN)
### 参考资料与资源
- [大佬整理的NERF介绍](https://github.com/yangjiheng/nerf_and_beyond_docs)  
- [原版 NERF 论文](https://arxiv.org/abs/2003.08934)  
   这是原版 NERF 的文章，看完我们可以讨论一下。
- [structure from motion(colmap)](https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)
- [nerf实现](https://github.com/yenchenlin/nerf-pytorch)



