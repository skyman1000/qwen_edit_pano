# GT2D可见几何实验

依据outputs/qwen_geometry_gt2d_run001/review/metrics.json、boxes.json和原始回答。12张图150对象，146个格式有效3D，4个未知。2D匹配100%是GT输入带来的，不是模型成绩。3D同类IoU≥.25仅2/150。

146个匹配对象中心误差：均值53.195m，中位数1.642m，P90=4.547m，7个超过10m。尺寸MAE=.609m。不能只引用被离群值支配的均值，也不能忽略离群值。

具体证据：8194nk5LbLH_ae91518ed77047b3bdeeca864cd04029/obs_0012的成功retry把2D框前三个值681.640625,679.6875,736.328125原样作为3D中心；zsNo4HB9uLZ_38e0c09ac7a748dbadea6471861b30c3/obs_0001原始回答中心[984,527.6,2.12]。这些值已经出现在模型回答中，不是Local转换引入。格式有效不等于物理有效，不修剪或任意缩放这些坐标。

本次GT框含较长的小数坐标，不宜据此直接断言Qwen所有3D能力失效。唯一后续对照：保持整张图片、GT目标类别、同一GT框、模型、内参和输出约定；将目标框画成红色标注，不在提示中写2D坐标数字。不裁剪，不改相机，不改隐藏补全。

新增run_simple_world.sh observe-marked入口。新输出目录必须与旧实验分开。结果若仍投影/尺度不可靠，停止这一路径的提示实验，Qwen保留语义和2D职责，转向有深度依据的几何估计。深度只能提供可见表面，不等价于完整物体AABB；隐藏补全仍未解决。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360
export SIMPLE_WORLD_RUN=qwen_edit_pano/outputs/qwen_geometry_marked_run001
export SIMPLE_WORLD_SOURCE=qwen_edit_pano/data/qwen_gt2d_probe_val12
bash qwen_edit_pano/world_pipeline/completion/run_simple_world.sh observe-marked
/data-nfs/gpu1-2/u13529658780/.conda/envs/qwen3vl/bin/python \
  -m qwen_edit_pano.world_pipeline.completion.qwen_upstream_review \
  --bundle qwen_edit_pano/data/qwen_upstream_val12_v1 \
  --observer qwen_edit_pano/outputs/qwen_geometry_marked_run001/observer \
  --output qwen_edit_pano/outputs/qwen_geometry_marked_run001/review
```
