level="$1"

echo $level
if [ "$level" -eq "1" ]; then
	echo Running Level 1
	th main.lua -fineWidth 32 -fineHeight 24 \
			-level 1 \
			-LR 4e-6 \
			-netType volcon -batchSize 32 \
			-cache checkpoint1.1 -augment 1 \
			-epochSize 1000 -optimizer adam \
			-retrain models/modelL1_3.t7
fi

if [ "$level" -eq "2" ]
then
	echo Running Level 2
	th main.lua -fineWidth 64 -fineHeight 48 \
			-level 2 \
			-LR 2e-6 \
			-netType volcon -batchSize 32 \
			-epochSize 1000 -optimizer adam \
			-cache checkpoint2.4 -augment 1 \
			-retrain models/modelL2_3.t7
fi

if [ "$level" -eq "3" ]
then
	echo Running Level 3
	th main.lua -fineWidth 128 -fineHeight 96 \
			-level 3 \
			-LR 1e-4 \
			-netType volcon -batchSize 32 \
			-epochSize 1000 -optimizer adam \
			-cache checkpoint3.2 -augment 1\
			-retrain models/modelL3_3.t7
fi

if [ "$level" -eq "4" ]
then
	echo Running Level 4
	th main.lua -fineWidth 256 -fineHeight 192 \
			-level 4 \
			-LR 2e-6 \
			-netType volcon -batchSize 32 \
			-epochSize 1000 -optimizer adam \
			-cache checkpoint4.2 -augment 1\
			-retrain models/modelL4_3.t7
fi

if [ "$level" -eq "5" ]
then
	echo Running Level 5
	th main.lua -fineWidth 512 -fineHeight 384 \
			-level 5 \
			-LR 1e-5 \
			-netType volcon -batchSize 16 \
			-epochSize 1000 -optimizer adam \
			-cache checkpoint5.1 -augment 0\
			-retrain models/modelL5_3.t7
fi
