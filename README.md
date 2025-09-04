## Dataset
**Adult**: [IBM/sensitive-subspace-robustness](https://github.com/IBM/sensitive-subspace-robustness).  
**Celeb**: Place the CelebA dataset (```list_attr_celeba.txt```, ```list_eval_partition.txt```, ```img_align_celeba```) under directory ```./celeba``` and run ```data_processing.py``` to process the dataset.   
## Run
Run the experiment:  
**Adult:**   
```
python main.py --method mixup/erm --mode dp --lam 0.5
```
**Celeb:**

``` 
python main_dp.py --method mixup/erm --lam 1.0 --target_id 2/31/33
```
## Recommended Lambda
Adult: 0.1 ~ 1.0 
Celeba: 1.0 ~ 5.0

## References
https://github.com/chingyaoc/fair-mixup
