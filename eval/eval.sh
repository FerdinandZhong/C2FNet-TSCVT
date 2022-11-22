# CUDA_VISIBLE_DEVICES=0 python main.py --methods s2+s0 --save_dir ./results --root_dir ~/PycharmProjects/TSCVT_code/eval --datasets CAMO+CHAMELEON+COD10K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods full --save_dir /root/Projects/C2FNet-TSCVT/eval/results/full/ --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/Projects/C2FNet-TSCVT/results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods BasicC2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/BasicC2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods BasicACFMC2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/BasicACFMC2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods BasicCIMC2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/BasicCIMC2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods BasicDGCMC2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/BasicDGCMC2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods BasicACFMDGCMC2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/BasicACFMDGCMC2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods C2FNetWOMSCA --save_dir /root/Projects/C2FNet-TSCVT/eval/results/C2FNetWOMSCA/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

# CUDA_VISIBLE_DEVICES=0 python main.py --methods C2FNet --save_dir /root/Projects/C2FNet-TSCVT/eval/results/C2FNet/  --gt_dir /root/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /root/autodl-tmp/C2FNet_results/ --datasets NC4K

CUDA_VISIBLE_DEVICES=0 run6 python main.py --methods BasicC2FNet --save_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/eval/results/BasicC2FNet  --gt_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/results --datasets NC4K

CUDA_VISIBLE_DEVICES=0 run6 python main.py --methods C2FNet --save_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/eval/results/C2FNet  --gt_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/data/TestDataset --pred_dir /export/home2/qishuai/Projects/C2FNet-TSCVT/results --datasets NC4K
