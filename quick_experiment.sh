epoch=500

mkdir results/
mkdir results/cifar_lenet

for seed in {0,} ; do
    mkdir results/cifar_lenet/${seed}
    
    for lr in {0.005,0.001,0.0005} ; do
	for class in {4,} ; do
	    mkdir results/cifar_lenet/${seed}/class_${class}
	    
	    for method in {decentlam,momentum_tracking,dsgdm,qg_dsgdm} ; do
		log_path=./results/cifar_lenet/${seed}/class_${class}/${method}_lr_${lr}/
		mkdir ${log_path}
		python evaluate_cifar.py ${method} ${log_path} --seed ${seed} --port 1579067 --nw config/ring_8/class${class}_${seed}.json --lr ${lr} --epoch ${epoch}
	    done
	   
	done
    done
done
