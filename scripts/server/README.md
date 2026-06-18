# Server sync & run (AutoDL, single RTX 3080)

Dev loop: edit locally in /home/rick/geta -> commit -> `git push fork geta-yolo26-pruning`
On server: `git -C ~/geta pull fork geta-yolo26-pruning`

First-time server clone:
    cd ~ && git clone -b geta-yolo26-pruning https://github.com/Bovey0809/geta.git
    cd ~/geta

All experiment entrypoints live under experiments/geta_yolo26/ and scripts/server/.
Run order: setup_env.sh -> link_coco.sh -> baseline -> sanity (test_yolo26) -> smoke -> full -> profile.
