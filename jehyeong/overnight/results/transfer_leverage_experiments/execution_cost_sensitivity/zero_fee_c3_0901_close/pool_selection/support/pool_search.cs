
using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
public class PoolSearch {
 class Row {
  public ulong Mask; public int K; public double CAGR, DD, Calmar, Abs, Signed, Weekly;
 }
 static int N,T,G; static int[][] Groups; static double[][] E,Work;
 static double[,] Corr,Weekly; static bool[] Positive; static int[] Picked;
 static List<Row>[] Top,Both; static Row[] Low,LowPositive,LowAllPositive;
 static long[] Counts; static long Leaves; static ulong Mandatory;
 static string Progress;
 static Row Make(ulong mask,int k,double ca,double dd,double ab,double si,double we) {
  return new Row {Mask=mask,K=k,CAGR=ca,DD=-dd,Calmar=dd>0?ca/dd:double.PositiveInfinity,
   Abs=k>1?ab/(k*(k-1)/2.0):double.NaN,Signed=k>1?si/(k*(k-1)/2.0):double.NaN,
   Weekly=k>1?we/(k*(k-1)/2.0):double.NaN};
 }
 static void Push(List<Row> list,Row row,int limit) {
  if(list.Count==limit && row.Calmar<=list[list.Count-1].Calmar) return;
  int at=0;while(at<list.Count && list[at].Calmar>=row.Calmar) at++;
  list.Insert(at,row); if(list.Count>limit)list.RemoveAt(limit);
 }
 static void Visit(int depth,int k,ulong mask,double[] sum,double ab,double si,double we,bool allPositive) {
  if(depth==G) {
   if(k==0)return;
   Leaves++;Counts[k]++;
   double ca=Math.Pow(sum[T-1]/k,252.0/T)-1.0;
   double peak=k,dd=0;
   for(int t=0;t<T;t++) {double v=sum[t];if(v>peak)peak=v;double d=1.0-v/peak;if(d>dd)dd=d;}
   Row row=Make(mask,k,ca,dd,ab,si,we);
   Push(Top[k],row,20);
   if((mask & Mandatory)==Mandatory)Push(Both[k],row,10);
   if(k>=2) {
    if(Low[k]==null || row.Abs<Low[k].Abs)Low[k]=row;
    if(ca>0 && (LowPositive[k]==null || row.Abs<LowPositive[k].Abs))LowPositive[k]=row;
    if(allPositive && (LowAllPositive[k]==null || row.Abs<LowAllPositive[k].Abs))LowAllPositive[k]=row;
   }
   if(Leaves%500000==0) {
    string msg="evaluated="+Leaves.ToString(CultureInfo.InvariantCulture);
    File.WriteAllText(Progress,msg);Console.WriteLine(msg);
   }
   return;
  }
  Visit(depth+1,k,mask,sum,ab,si,we,allPositive);
  foreach(int id in Groups[depth]) {
   double a=ab,s=si,w=we;
   for(int q=0;q<k;q++) {double c=Corr[id,Picked[q]];a+=Math.Abs(c);s+=c;w+=Math.Abs(Weekly[id,Picked[q]]);}
   double[] next=Work[depth]; double[] add=E[id];
   for(int t=0;t<T;t++)next[t]=sum[t]+add[t];
   Picked[k]=id;
   Visit(depth+1,k+1,mask|(1UL<<id),next,a,s,w,allPositive && Positive[id]);
  }
 }
 static string F(double v) {return v.ToString("R",CultureInfo.InvariantCulture);}
 static void Write(StreamWriter writer,string kind,Row r) {
  if(r==null)return;
  writer.WriteLine(kind+","+r.Mask+","+r.K+","+F(r.CAGR)+","+F(r.DD)+","+F(r.Calmar)+","+F(r.Abs)+","+F(r.Signed)+","+F(r.Weekly));
 }
 public static void Main(string[] args) {
  Progress=args[3];
  using(BinaryReader b=new BinaryReader(File.OpenRead(args[0]))) {
   N=b.ReadInt32();T=b.ReadInt32();G=b.ReadInt32();Mandatory=b.ReadUInt64();
   Groups=new int[G][];
   for(int g=0;g<G;g++){Groups[g]=new int[b.ReadInt32()];for(int j=0;j<Groups[g].Length;j++)Groups[g][j]=b.ReadInt32();}
   E=new double[N][];for(int i=0;i<N;i++){E[i]=new double[T];for(int t=0;t<T;t++)E[i][t]=b.ReadDouble();}
   Corr=new double[N,N];Weekly=new double[N,N];
   for(int i=0;i<N;i++)for(int j=0;j<N;j++)Corr[i,j]=b.ReadDouble();
   for(int i=0;i<N;i++)for(int j=0;j<N;j++)Weekly[i,j]=b.ReadDouble();
   Positive=new bool[N];for(int i=0;i<N;i++)Positive[i]=b.ReadInt32()!=0;
  }
  Work=new double[G][];for(int i=0;i<G;i++)Work[i]=new double[T];
  Picked=new int[G];Counts=new long[G+1];Top=new List<Row>[G+1];Both=new List<Row>[G+1];
  Low=new Row[G+1];LowPositive=new Row[G+1];LowAllPositive=new Row[G+1];
  for(int i=0;i<=G;i++){Top[i]=new List<Row>();Both[i]=new List<Row>();}
  Visit(0,0,0UL,new double[T],0,0,0,true);
  using(StreamWriter w=new StreamWriter(args[1])) {
   w.WriteLine("kind,mask,accounts,CAGR,MDD,Calmar,mean_abs_corr,mean_signed_corr,mean_abs_weekly_corr");
   for(int k=1;k<=G;k++) {
    foreach(Row r in Top[k])Write(w,"TOP_CALMAR",r);
    foreach(Row r in Both[k])Write(w,"TOP_CALMAR_WITH_KP_KD",r);
    Write(w,"MIN_ABS_CORR",Low[k]);Write(w,"MIN_ABS_CORR_POSITIVE_POOL",LowPositive[k]);
    Write(w,"MIN_ABS_CORR_ALL_POSITIVE",LowAllPositive[k]);
   }
  }
  using(StreamWriter w=new StreamWriter(args[2])) {
   w.WriteLine("accounts,combinations");for(int k=1;k<=G;k++)w.WriteLine(k+","+Counts[k]);
  }
  File.WriteAllText(Progress,"complete="+Leaves);Console.WriteLine("complete="+Leaves);
 }
}
