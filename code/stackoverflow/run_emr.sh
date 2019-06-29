# Launch AWS EMR cluster configured to use iPython 3.6/with pip-3.6 modules installed
aws emr create-cluster --applications Name=Hadoop Name=Spark \
    --ec2-attributes '{"KeyName":"agile_data_science","InstanceProfile":"EMR_EC2_DefaultRole","SubnetId":"subnet-1f975b56","EmrManagedSlaveSecurityGroup":"sg-0eee65e83f2ab9476","EmrManagedMasterSecurityGroup":"sg-07357e0e7ee9315be"}' \
    --release-label emr-5.24.1 \
    --log-uri 's3n://aws-logs-087121496299-us-west-2/elasticmapreduce/' \
    --instance-groups '[{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":1000,"VolumeType":"gp2"},"VolumesPerInstance":1}],"EbsOptimized":true},"InstanceGroupType":"MASTER","InstanceType":"m5.2xlarge","Name":"Master - 1"},{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":2048,"VolumeType":"gp2"},"VolumesPerInstance":1}],"EbsOptimized":true},"InstanceGroupType":"CORE","InstanceType":"r5.4xlarge","Name":"Core - 2"},{"InstanceCount":6,"BidPrice":"OnDemandPrice","EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":2048,"VolumeType":"gp2"},"VolumesPerInstance":1}],"EbsOptimized":true},"InstanceGroupType":"TASK","InstanceType":"r5.4xlarge","Name":"Task - 3"}]' \
    --configurations '[{"Classification":"spark-env","Properties":{},"Configurations":[{"Classification":"export","Properties":{"PYSPARK_PYTHON":"/usr/local/bin/ipython"}}]}]' \
    --auto-scaling-role EMR_AutoScaling_DefaultRole \
    --bootstrap-actions '[{"Path":"s3://data-syndrome/emr_bootstrap.sh","Name":"Python Setup"}]' \
    --ebs-root-volume-size 100 \
    --service-role EMR_DefaultRole \
    --enable-debugging \
    --name 'Deep Products Stackoverflow ETL Cluster' \
    --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
    --region us-west-2
