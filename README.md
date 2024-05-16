Cloned the MLD git repo: https://github.com/chenfengye/motion-latent-diffusion ### 1. Conda environment```conda create python=3.9 --name mldconda activate mld```Install the packages in `requirements.txt` and install [PyTorch 2.2.2](https://pytorch.org/)If any package error, refer packages_ref.txt file```pip install -r requirements.txt```We test our code on Python 3.9.12 and PyTorch 2.2.2### 2. DependenciesRun the script to download dependencies materials:```bash prepare/download_smpl_model.shbash prepare/prepare_clip.sh```For Text to Motion Evaluation```bash prepare/download_t2m_evaluators.sh```### 3. Prepare the HumanML datasetPlease refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for text-to-motion dataset setup.You can download the Human ML data from our drive : https://drive.google.com/file/d/1bOmzxuH8xNk2XM4Onfde0tstDovgRowV/view?usp=sharing. However, note that this is Due to the distribution policy of AMASS dataset, we are not allowed to distribute the data directly. We provided just for this project to reduce the effort in settingup the data and replicate the results## 4. Our Code Changes (and explaining the folder structure)datasets    - humanml3d: place the downloaded dataset here configs: contains the arguments    - assets (modified this file to adjust the paths accordingly): path configs        - config_vae_humanml3d.yaml (added this file): human ml 3d config file for VAE training        - config_GAN_humanml3d.yaml (added this file): human ml 3d config file for GAN trainingprepare:     - contains bash scripts to download the dependecies. Also make sure once you download edit the path accordingly in the configs/assest.yaml mld    - models:             -get_model.py: edited the line 6 in this file to handle modeltype GAN and WGAN                 -model_type:             - base.py: base pytorch lighting module, mld.package            - GAN.py (added this file): this files used in the train.py. Internally calls architectures/gan_arcitecture.py etc            - WGAN.py (added this file):  this files used in the train.py. Internally calls architectures/wgan_arcitecture.py etc            -architectures: define architectures            - gan_arcitecture.py (added this file): Has the simple GAN architecture with BCE loss            - wgan_arcitecture.py (added this file):  Has the simple GAN architecture with Wassestein loss            - gan_dense.py (added this file):  Has the Dense GAN architecture with BCE loss            - wgan_dense.py (added this file): Has the Dense GAN architecture with Wassestein loss            - style_gan.py (added this file): Has the style GAN architecture with BCE loss            - wstyle_gan.py (added this file): Has the style GAN architecture with Wassestein loss    train.py  -> file in which training VAE or GAN happens. trainer_bash.sh (added this file) -> To run the training in GPUtest.py -> testing and calculating the evaluation metricsdemo.py - loading the trained models and showing demo (text to motion)     render.py -> visulaize motions using blenderdemo/example.txt: text input for testing    results:    - GAN: results with GAN architectures        - WGAN: results with WGAN architecture      ### 5. Train VAEPlease first check the parameters in `configs/config_vae_humanml3d.yaml`, e.g. `NAME`Then, run the following command:```python -m train --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug```### 6. Train GAN        ### Ready to train GAN modelPlease update the parameters in `configs/config_mld_kitml.yaml`, e.g. `NAME`, Update the `PRETRAINED_VAE` to the `latest VAE ckpt model path` in previous stepUse model_type=GAN for GAN training, else use WGAN for Wassestein GAN training. Also in the model_type/GAN.py or model_type/WGAN.py from lines from 77-79, uncomment only the required self.gan initialization based on the required architecture of GAN (simple, dense, style)Then, run the following command:```python -m train --cfg configs/config_GAN_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug```    ### 7. Test or Evaluate the modelPlease first put the tained model checkpoint path to `TEST.CHECKPOINT` in `configs/config_GAN_humanml3d.yaml`.Then, run the following command:```To test trained VAE: python -m test --cfg configs/config_vae_humanml3d.yaml --cfg_assets configs/assets.yamlTo test trained GAN: python -m test --cfg configs/config_GAN_humanml3d.yaml --cfg_assets configs/assets.yaml```### 8. DemoDownload the GAN pretrained and VAE models from: https://drive.google.com/drive/folders/1DiiECLLP7wXv6Kh1RRAaO5ZJEwNIC1hX?usp=sharingEdit the `TEST.CHECKPOINT` and `PRETRAINED_VAE` path in configs/config_GAN_humanml3d.yaml accordingly.To run the demo using the trained GAN for the inputs demo/example.txt```Run demo using:  python demo.py --cfg configs/config_GAN_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/example.txt```The outputs:- `npy file`: the generated motions with the shape of (nframe, 22, 3) for HumanML. You can find these in the results section. - `text file`: the input text prompt</details>### 9. BlenderYou can find the npy converted to video here: https://drive.google.com/drive/folders/1Ik9CkRPsKm3_Gy8cMDZSJg1_K-qC8YDC?usp=drive_link