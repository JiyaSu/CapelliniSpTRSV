
CSV="result.csv"
FIN="finish"
FILE="metrics"

i=0
#rm $CSV
mkdir $FILE
while read line
do
    line2=${line%?}
    echo nvprof  --metrics stall_exec_dependency,stall_sync --csv --log-file metrics/$i.csv ./main $line2
    nvprof  --metrics stall_exec_dependency,stall_sync --csv --log-file metrics/$i.csv ./main $line2
    i=`expr $i + 1`
done < name.txt

echo $FIN


