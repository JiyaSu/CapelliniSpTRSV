MTXADDRESS="/home/storage4TB/forJiya"
PROGRAMADDRESS="/home/sujiya/SpTRSV/2_SyncFree_csc2"
PROGRAM="./main"
CSV="result.csv"
FIN="finish"

#rm $CSV
#cd $MTXADDRESS
#a=`ls`
##echo $a
#
#for i in $a
#do
#    cd $MTXADDRESS/$i
#    b=`ls`
#    for j in $b
#    do
#        cd $PROGRAMADDRESS
#        echo $PROGRAM $MTXADDRESS/$i/$j
#        $PROGRAM $MTXADDRESS/$i/$j
#    done
#done
#echo $FIN


rm $CSV
cd $MTXADDRESS
a=`ls`
#echo $a
start=0
jump=0
for i in $a
do
cd $MTXADDRESS/$i
b=`ls`
for j in $b
do
cd $PROGRAMADDRESS
#echo $PROGRAM $MTXADDRESS/$i/$j/$j
if [ $j = "mawi_201512012345.mtx" ] || [ $j = "mawi_201512020000.mtx" ] || [ $j = "mawi_201512020030.mtx" ] || [ $j = "mawi_201512020130.mtx" ] || [ $j = "mawi_201512020330.mtx" ] || [ $j = "sk-2005.mtx" ]
then
jump=1
else
echo $PROGRAM $MTXADDRESS/$i/$j
$PROGRAM $MTXADDRESS/$i/$j
fi
done
done
echo $FIN
