
CSV="result.csv"
FIN="finish"
FILE="metrics"

i=0
rm $CSV
mkdir $FILE
while read line
do
    line2=${line%?}
    echo nvprof --kernels "sptrsv_syncfree_cuda_executor_update"  --metrics dram_read_throughput,dram_write_throughput --csv --log-file metrics/$i.csv ./main $line2
    nvprof --kernels "sptrsv_syncfree_cuda_executor_update"  --metrics dram_read_throughput,dram_write_throughput --csv --log-file metrics/$i.csv ./main $line2
    i=`expr $i + 1`
done < name.txt

echo $FIN


