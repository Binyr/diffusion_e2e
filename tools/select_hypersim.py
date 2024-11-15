from pathlib import Path

def hypersim_rgb_path():
    paths = [
        "data/hypersim/processed/train/ai_008_009/rgb_cam_00_fr0000.png",
        "data/hypersim/processed/test/ai_001_010/rgb_cam_00_fr0000.png",
        "data/hypersim/processed/train/ai_002_001/rgb_cam_00_fr0000.png",
        "data/hypersim/processed/test/ai_005_001/rgb_cam_00_fr0000.png",
        "data/hypersim/processed/val/ai_007_001/rgb_cam_00_fr0000.png"
    ]
    save_dir = Path("data/hypersim_selected")
    import shutil
    for i, path in enumerate(paths):
        shutil.copyfile(path, str(save_dir / f"{i}.png"))
    
if __name__ == "__main__":
    hypersim_rgb_path()