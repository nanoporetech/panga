Search.setIndex({docnames:["index","modules","panga","panga.accumulate","panga.algorithm","panga.classify","panga.conclude","panga.data","panga.data.config","panga.metric","panga.split","panga.stage"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:54},filenames:["index.rst","modules.rst","panga.rst","panga.accumulate.rst","panga.algorithm.rst","panga.classify.rst","panga.conclude.rst","panga.data.rst","panga.data.config.rst","panga.metric.rst","panga.split.rst","panga.stage.rst"],objects:{"":{panga:[2,0,0,"-"]},"panga.accumulate":{base:[3,0,0,"-"],blocking_report:[3,0,0,"-"],simple:[3,0,0,"-"],visualize:[3,0,0,"-"]},"panga.accumulate.base":{AccumulateBase:[3,1,1,""]},"panga.accumulate.base.AccumulateBase":{finalize:[3,2,1,""],process_read:[3,2,1,""],process_reads:[3,2,1,""]},"panga.accumulate.blocking_report":{BlockingReport:[3,1,1,""]},"panga.accumulate.blocking_report.BlockingReport":{finalize:[3,2,1,""],generate_plots:[3,2,1,""],make_bar_chart:[3,3,1,""],make_stacked_duty_time:[3,3,1,""],plot_exp_hist:[3,3,1,""]},"panga.accumulate.simple":{ChannelReport:[3,1,1,""],ClassicMetricSummary:[3,1,1,""],CountClasses:[3,1,1,""],FilteredMetricSummary:[3,1,1,""],MetricSummary:[3,1,1,""]},"panga.accumulate.simple.ChannelReport":{finalize:[3,2,1,""]},"panga.accumulate.simple.CountClasses":{finalize:[3,2,1,""]},"panga.accumulate.simple.FilteredMetricSummary":{finalize:[3,2,1,""]},"panga.accumulate.visualize":{DutyTimeDistPlot:[3,1,1,""],DutyTimePlot:[3,1,1,""],EventRatePlot:[3,1,1,""]},"panga.accumulate.visualize.DutyTimeDistPlot":{finalize:[3,2,1,""]},"panga.accumulate.visualize.DutyTimePlot":{finalize:[3,2,1,""]},"panga.accumulate.visualize.EventRatePlot":{finalize:[3,2,1,""]},"panga.algorithm":{deltasplit:[4,0,0,"-"],read_bounds_from_delta:[4,0,0,"-"]},"panga.algorithm.deltasplit":{read_bounds_from_delta:[4,4,1,""]},"panga.algorithm.read_bounds_from_delta":{read_bounds_from_delta:[4,4,1,""]},"panga.classify":{base:[5,0,0,"-"],read_rules:[5,0,0,"-"],simple:[5,0,0,"-"]},"panga.classify.base":{ClassifyBase:[5,1,1,""]},"panga.classify.base.ClassifyBase":{n_reads:[5,5,1,""],process_read:[5,2,1,""],process_reads:[5,2,1,""],requires:[5,5,1,""]},"panga.classify.read_rules":{Key:[5,1,1,""],ReadRules:[5,1,1,""],Rule:[5,1,1,""],SubRule:[5,1,1,""]},"panga.classify.read_rules.Key":{keyword:[5,5,1,""],position:[5,5,1,""]},"panga.classify.read_rules.ReadRules":{n_reads:[5,5,1,""],requires:[5,5,1,""]},"panga.classify.read_rules.Rule":{evaluate:[5,2,1,""],requires:[5,5,1,""],requires_left:[5,5,1,""],requires_right:[5,5,1,""]},"panga.classify.read_rules.SubRule":{evaluate:[5,2,1,""],requires:[5,5,1,""],requires_left:[5,5,1,""],requires_right:[5,5,1,""]},"panga.classify.simple":{DeltaClassifier:[5,1,1,""],MetricClassifier:[5,1,1,""],SimpleClassifier:[5,1,1,""],StandardClassifier:[5,1,1,""]},"panga.classify.simple.DeltaClassifier":{n_reads:[5,5,1,""],requires:[5,5,1,""]},"panga.classify.simple.MetricClassifier":{requires:[5,5,1,""]},"panga.classify.simple.SimpleClassifier":{n_reads:[5,5,1,""],requires:[5,5,1,""]},"panga.classify.simple.StandardClassifier":{n_reads:[5,5,1,""],requires:[5,5,1,""]},"panga.cmdargs":{ExpandRanges:[2,1,1,""],FileExists:[2,1,1,""]},"panga.conclude":{base:[6,0,0,"-"],simple:[6,0,0,"-"]},"panga.conclude.base":{ConcludeBase:[6,1,1,""]},"panga.conclude.base.ConcludeBase":{finalize:[6,2,1,""],process_read:[6,2,1,""],process_reads:[6,2,1,""]},"panga.conclude.simple":{Fast5Write:[6,1,1,""]},"panga.conclude.simple.Fast5Write":{process_read:[6,2,1,""]},"panga.data":{config:[8,0,0,"-"]},"panga.fileio":{file_has_fields:[2,4,1,""],read_chunks:[2,4,1,""],readchunkedtsv:[2,4,1,""],readtsv:[2,4,1,""],take_a_peak:[2,4,1,""]},"panga.iterators":{blocker:[2,4,1,""],empty_iterator:[2,4,1,""],window:[2,4,1,""]},"panga.metric":{base:[9,0,0,"-"],metrics:[9,0,0,"-"],simple:[9,0,0,"-"]},"panga.metric.base":{MetricBase:[9,1,1,""]},"panga.metric.base.MetricBase":{process_read:[9,2,1,""],process_reads:[9,2,1,""],provides:[9,5,1,""],register_channel_metric:[9,2,1,""],register_event_metric:[9,2,1,""],register_event_multi_metric:[9,2,1,""],register_read_meta_metric:[9,2,1,""],requires_events:[9,5,1,""],requires_meta:[9,5,1,""],requires_raw:[9,5,1,""]},"panga.metric.metrics":{apply_to_field:[9,4,1,""],duration:[9,4,1,""],entropy:[9,4,1,""],filled_window:[9,4,1,""],get_channel_attr:[9,4,1,""],get_channel_attr_at_read:[9,4,1,""],get_read_meta_attr:[9,4,1,""],is_multiple_at_read:[9,4,1,""],is_off_at_read:[9,4,1,""],is_saturated_at_read:[9,4,1,""],locate_stall:[9,4,1,""],med_mad_data:[9,4,1,""],n_active_flicks_in_read:[9,4,1,""],n_global_flicks_in_read:[9,4,1,""],range_data:[9,4,1,""],read_start_time:[9,4,1,""],sliding_metric:[9,4,1,""],threshold_changepoint:[9,4,1,""],windowed_metric:[9,4,1,""]},"panga.metric.simple":{BlockingMetrics:[9,1,1,""],MockMetrics:[9,1,1,""],SimpleMetrics:[9,1,1,""],SolidStateMetrics:[9,1,1,""],StandardMetrics:[9,1,1,""],StandardMinionMetrics:[9,1,1,""],StandardMinknowMetrics:[9,1,1,""],SummaryAndStandardMetrics:[9,1,1,""],SummaryMetrics:[9,1,1,""]},"panga.read_builder":{MoreHelpAction:[2,1,1,""],ParseJsonDictAction:[2,1,1,""],SetReadRulesAction:[2,1,1,""],accumulate_channels:[2,4,1,""],except_functor:[2,4,1,""],get_accumulators:[2,4,1,""],get_argparser:[2,4,1,""],get_class_arg_store:[2,4,1,""],get_class_args:[2,4,1,""],get_classifier:[2,4,1,""],get_concluders:[2,4,1,""],get_metrifier:[2,4,1,""],get_pipeline:[2,4,1,""],get_second_stage:[2,4,1,""],get_splitter:[2,4,1,""],load_yaml_config:[2,4,1,""],main:[2,4,1,""],make_resolver:[2,4,1,""],process_args:[2,4,1,""],process_channel:[2,4,1,""],process_channel_queue:[2,4,1,""],resolve_rules_config:[2,4,1,""]},"panga.split":{base:[10,0,0,"-"],simple:[10,0,0,"-"]},"panga.split.base":{Read:[10,1,1,""],SplitBase:[10,1,1,""]},"panga.split.base.Read":{raw:[10,5,1,""]},"panga.split.base.SplitBase":{meta_keys:[10,5,1,""],provides_events:[10,5,1,""],provides_raw:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple":{AdaptiveThreshold:[10,1,1,""],DeltaSplit:[10,1,1,""],Fast5Split:[10,1,1,""],FixedInterval:[10,1,1,""],RandomData:[10,1,1,""],SummarySplit:[10,1,1,""]},"panga.split.simple.AdaptiveThreshold":{meta_keys:[10,5,1,""],provides_events:[10,5,1,""],provides_raw:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple.DeltaSplit":{meta_keys:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple.Fast5Split":{load_fast5_meta:[10,2,1,""],meta_keys:[10,5,1,""],provides_events:[10,5,1,""],provides_raw:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple.FixedInterval":{meta_keys:[10,5,1,""],provides_events:[10,5,1,""],provides_raw:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple.RandomData":{meta_keys:[10,5,1,""],provides_events:[10,5,1,""],provides_raw:[10,5,1,""],reads:[10,2,1,""]},"panga.split.simple.SummarySplit":{iterate_input_file:[10,2,1,""],meta_keys:[10,5,1,""],reads:[10,2,1,""],requires_meta:[10,5,1,""]},"panga.stage":{base:[11,0,0,"-"],blocking:[11,0,0,"-"],simple:[11,0,0,"-"]},"panga.stage.base":{StageBase:[11,1,1,""]},"panga.stage.base.StageBase":{process_reads:[11,2,1,""]},"panga.stage.blocking":{BlockingAnalysis:[11,1,1,""]},"panga.stage.simple":{PassStage:[11,1,1,""]},"panga.util":{add_prefix:[2,4,1,""],ensure_dir_exists:[2,4,1,""],print_config_dir:[2,4,1,""]},panga:{accumulate:[3,0,0,"-"],algorithm:[4,0,0,"-"],classify:[5,0,0,"-"],cmdargs:[2,0,0,"-"],conclude:[6,0,0,"-"],data:[7,0,0,"-"],fileio:[2,0,0,"-"],iterators:[2,0,0,"-"],metric:[9,0,0,"-"],read_builder:[2,0,0,"-"],split:[10,0,0,"-"],stage:[11,0,0,"-"],util:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","staticmethod","Python static method"],"4":["py","function","Python function"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:staticmethod","4":"py:function","5":"py:attribute"},terms:{"boolean":[2,5],"class":[0,2,3,5,6,9,10,11],"const":2,"default":2,"final":[2,3,6],"function":[2,9],"int":9,"return":[2,5,9],"static":3,"true":[2,3,4,10],For:9,The:[0,5],Useful:5,Will:2,_after:5,_befor:5,abil:0,abnorm:9,abov:[0,9],absolut:9,accept:9,accord:5,accumul:[0,1,2],accumulate_channel:2,accumulatebas:3,across:[0,9],action:2,activ:[0,9],adapt:3,adaptivethreshold:10,add:6,add_prefix:2,added:2,addit:0,advanc:0,after:5,algorithm:[0,1,2],alia:5,all:[0,2,5],allow:0,analys:0,ani:[2,3,5],anoth:2,appli:[5,9],applic:0,apply_to_field:9,apt:0,arbitrari:0,arbitrarili:0,arg:[2,9],arg_stor:2,argpars:2,argstor:2,argument:[2,9],arr:9,arrai:[2,9],assign:0,associ:11,attribut:[5,9],axi:9,backward:0,base:[0,1,2],befor:[5,6],begin:9,below:9,bermuda:3,better:2,between:0,bias_voltage_chang:9,bin:0,bin_siz:3,black:3,block:[1,2],block_ch_report:3,block_channel_report:3,block_class:3,block_run_report:3,blocker:2,blocking_report:[1,2],blockinganalysi:11,blockingmetr:9,blockingreport:3,blue:3,bool:2,boundari:[0,4,9],build:5,bz2:2,calcul:[0,9],call:[3,5,6],can:[0,3,5],capture_rank:10,central:5,ch_result:3,chang:0,changepoint:9,channel:[0,2,3,6,9,10],channel_meta:[9,10],channel_report:3,channelreport:3,check:[2,5],choic:2,chunk:2,chunk_siz:2,class_arg:2,class_filt:[3,6],class_metr:5,classicmetricsummari:3,classif:0,classifi:[0,1,2,11],classifybas:5,cleanup:3,clear:0,client:3,cls:2,cls_arg:2,cmdarg:[0,1],col_prefix:3,colours_dict:3,column:9,combin:3,compar:5,comparison:5,complex:0,compon:2,comput:[5,9],concern:0,conclud:[0,1,2],concludebas:6,config:[0,2,7],config_dict:2,connect:3,consid:9,contain:[2,5,9],content:1,context_meta:10,contigu:[0,9],cooki:0,coordin:0,copi:[4,9],correspond:[0,2],could:[0,5],count:3,countclass:3,cours:0,creat:[2,9],current:[0,5,9],data:[0,1,2,3,9,10],decompress:2,defin:0,delta:[4,10],deltaclassifi:5,deltasplit:[1,2,10],depend:[0,9],dest:2,detect:[4,9],determin:[0,9],dev:0,deviat:9,dict:2,dictionari:[5,9],differ:0,direct:9,directori:[0,2],displai:2,doe:0,doesn:2,done:5,down:9,durat:9,dure:5,duty_class:3,duty_tim:3,duty_time_dist:3,dutytimedistplot:3,dutytimeplot:3,each:[0,3],either:[5,9],element:5,empti:2,empty_iter:2,end:[0,9],end_of_read:3,ensur:5,ensure_dir_exist:2,entri:[5,9],entropi:9,entrypoint:2,environ:0,estim:9,evalu:5,event:[0,4,9,10],event_r:3,eventrateplot:3,exampl:0,except:2,except_functor:2,exclud:0,exist:[2,5],exogen:0,expandrang:2,expect:[0,9],experi:0,extens:2,failur:9,fals:[2,3,10],fast5:[0,10],fast5_class_filt:6,fast5split:10,fast5writ:6,featur:9,fetch:9,field:[2,5,9],file:2,file_has_field:2,fileexist:2,filehandl:3,fileio:[0,1],filenam:[2,6],fill:9,filled_window:9,filter:3,filter_ch_mux:3,filter_class:3,filter_count:3,filter_dur:3,filter_sum_dur:3,filtered_summari:3,filteredmetricsummari:3,first:[0,2,9],fixed_threshold:10,fixedinterv:10,flick:9,fname:2,follow:0,found:0,framework:0,fresh:2,from:[2,3,4,5,9,10],func:9,func_to_appli:9,gener:0,generate_plot:3,genfromtxt:2,get:[0,2],get_accumul:2,get_argpars:2,get_channel_attr:9,get_channel_attr_at_read:9,get_class_arg:2,get_class_arg_stor:2,get_classifi:2,get_conclud:2,get_metrifi:2,get_pipelin:2,get_read_meta_attr:9,get_second_stag:2,get_splitt:2,given:[2,5,9],global:9,had:3,has:[2,5],have:9,head:2,header:2,help:2,here:0,high:9,higher:9,hist_bin:3,hope:0,identifi:[0,2],ignor:2,implement:0,incarn:0,index:[0,9],indic:[5,9],individu:0,inf:10,initi:2,initial_classif:5,input:[2,9],input_summari:10,interfac:[0,9],interv:10,is_multiple_at_read:9,is_off_at_read:9,is_satur:5,is_saturated_at_read:9,item:9,iter:[0,1,3,5,6,9],iterate_input_fil:10,json:2,kei:[0,5,6,9],keyword:[2,5],kwarg:[2,11],label:0,lawngreen:3,left:5,length:[2,9],less:3,like:[2,9],line:2,list:[2,5,9],liter:5,load:2,load_fast5_meta:10,load_yaml_config:2,locat:9,locate_stal:9,look_back:10,look_back_n:4,low:[0,9],lower:9,mad:9,mad_estimate_ev:9,mai:[0,2],main:2,make:[0,2],make_bar_chart:3,make_plot:3,make_resolv:2,make_stacked_duty_tim:3,manipul:[2,11],map:5,matrix:0,max_tim:10,maximum:9,mean:[5,9],meant:5,measur:9,med_mad_data:9,median:9,median_curr:5,memori:0,meta:[0,3,9,10],meta_kei:10,metavar:2,method:3,methodolog:0,metric:[0,1,2,3,5,6,11],metricbas:9,metricclassifi:5,metricsummari:3,metrifi:[2,5],min_count:3,min_sampl:9,minimum:[3,9],minknow:[0,10],miscellan:2,mockmetr:9,modul:[0,1],more:0,morehelpact:2,move:9,multidimension:9,multipl:[3,9],multiprocess:[0,2],mux:[3,9],mux_chang:9,my_compon:2,n_active_flicks_in_read:9,n_adapter_or_strand_to_first_bermuda:3,n_adapter_to_first_bermuda:3,n_bermuda:3,n_bin:3,n_chunk:2,n_global_flicks_in_read:9,n_line:2,n_read:[5,10],n_strand_to_first_bermuda:3,naiv:0,name:9,namespac:2,narg:2,nbin:3,ndarrai:9,necessari:2,need:[0,9],neg:9,neighbour:0,network:3,none:[2,3,5,6,9,10,11],notic:0,number:[2,5,9],numer:2,numpi:2,object:[3,5,6,9,10,11],obtain:10,occur:9,off:9,offset:9,oliv:3,one:[0,2],onli:[0,9],operand:5,option:[0,2,9],option_str:2,order:5,organis:2,origin:9,ossetra:4,other:2,out:3,outfil:3,outpath:[0,3,6,10],output:0,over:[0,9],packag:[0,1],pad:[2,9],page:0,pair:[0,6],panga_config_dir:0,panga_directori:0,param:9,paramet:[2,5,9],pars:[2,5],parsejsondictact:2,part1:5,part2:5,part:5,particip:5,pas:4,pass:5,passstag:11,path:2,path_to_bulk_hdf:0,pcnt_time_:3,per:2,perform:3,pipelin:5,pkg_resourc:2,plot_count_col:3,plot_exp_hist:3,plot_hist_col:3,plot_hist_sum_col:3,plot_path:3,png:3,point:[0,2,3],pore:[3,5],pore_befor:5,pore_rank:10,posit:[5,9],possibl:[0,2],post:9,post_process:3,postion:2,prefer:2,prefix:[2,3,6,10],prepar:2,preset:2,previou:0,print_config_dir:2,prior:9,process:[0,2,6,9],process_arg:2,process_channel:2,process_channel_queu:2,process_read:[3,5,6,9,11],produc:9,properli:3,provid:[0,9,10],provides_ev:10,provides_raw:10,pure:[5,9],put:2,python:0,queue:2,rais:2,randomdata:10,rang:9,range_data:9,rather:0,raw:[0,9,10],read:[2,3,5,6,9,10,11],read_bounds_from_delta:[1,2],read_build:1,read_chunk:2,read_length:10,read_meta:9,read_metr:[3,6],read_rul:[1,2],read_start_tim:9,read_summari:0,read_summary_filt:3,readchunkedtsv:2,readrul:[2,5],readtsv:2,receiv:5,recovered_class:5,red:3,redetect:0,refer:5,region:9,regist:9,register_channel_metr:9,register_event_metr:9,register_event_multi_metr:9,register_read_meta_metr:9,report:3,report_class:3,repres:5,requir:[0,2,3,5,9],requires_ev:9,requires_left:5,requires_meta:[9,10],requires_raw:9,requires_right:5,resolv:2,resolve_rules_config:2,resourc:3,result:[2,5,9],results_dict:3,retriev:9,right:5,rolling_window:10,rule:[2,5],rule_nam:5,rule_str:5,run:2,run_result:3,same:0,satur:[0,3,9],search:0,see:[2,9],self:3,separ:0,set:[2,3,5,9],set_bias_voltag:9,setreadrulesact:2,share:2,should:[0,5,9],simpl:[0,1,2],simpleclassifi:5,simplemetr:9,simpli:9,singl:[3,6,9],size:2,slide:2,sliding_metr:9,smart:0,solidstatemetr:9,some:0,somewhat:0,sourc:[0,2,3,4,5,6,9,10,11],specif:2,specifi:9,split:[0,1,2],splitbas:10,splitter:[2,9,11],stage:[0,1,2],stagebas:11,stall:9,standard_minknow_class:0,standardclassifi:5,standardmetr:9,standardminionmetr:9,standardminknowmetr:9,start:[0,9],start_tim:9,state:[0,9],state_class:5,static_class:3,statist:0,statu:0,step:0,store:[0,9],str:2,strand:[3,5],strand_detector:4,stream:11,string:5,structur:[5,9],stuff:2,submodul:[0,1],subpackag:[0,1],subrul:5,success:2,sudo:0,sum_duration_adapter_to_first_bermuda:3,sum_duration_pore_to_first_bermuda:3,sum_duration_productive_to_first_bermuda:3,sum_duration_strand_to_first_bermuda:3,sum_duration_unproductive_to_first_bermuda:3,summari:0,summaris:9,summary_fil:[0,3],summaryandstandardmetr:9,summarymetr:9,summarysplit:10,sure:2,surround:[0,5],system:0,take:0,take_a_peak:2,test:2,than:[0,3],thei:5,them:5,thi:[0,2,3,9],those:5,three:0,thresh_factor:10,threshold:9,threshold_changepoint:9,throughout:0,time:[2,3,9],time_bin:3,time_sinc:5,time_to_first_bermuda:3,to_inject:5,total:9,traceback:2,tracking_meta:10,translat:2,tsv:2,tupl:[2,5,9],two:9,txt:[0,3],type:2,ubuntu:0,ultim:0,under:[2,9],unproduct:3,updat:[5,9],use:[0,2,5,9],use_cython:4,used:[0,2,5,9],useful:0,user1:5,using:[2,4,5],util:[0,1],valu:[0,2,6,9],variou:10,venv:0,virtual:0,virtualenv:0,visual:[1,2],well_id:9,were:3,when:[2,5],where:[0,5,9],whether:9,which:[3,9],width:9,window:[2,9],window_bound:9,window_length:9,window_start:9,windowed_metr:9,with_ev:10,with_raw:10,with_stat:10,within:9,without:5,worker:2,would:0,wrapper:2,write:3,x_label:3,y_label:3,yaml:2,yaml_conf:2,yield:[2,9,10],yml:0,you:0,your:0},titles:["Welcome to Panga\u2019s documentation!","panga","panga package","panga.accumulate package","panga.algorithm package","panga.classify package","panga.conclude package","panga.data package","panga.data.config package","panga.metric package","panga.split package","panga.stage package"],titleterms:{accumul:3,algorithm:4,api:0,base:[3,5,6,9,10,11],block:11,blocking_report:3,builder:0,classifi:5,cmdarg:2,conclud:6,config:8,content:[0,2,3,4,5,6,7,8,9,10,11],data:[7,8],deltasplit:4,document:0,fileio:2,full:0,indic:0,instal:0,iter:2,metric:9,modul:[2,3,4,5,6,7,8,9,10,11],packag:[2,3,4,5,6,7,8,9,10,11],panga:[0,1,2,3,4,5,6,7,8,9,10,11],read:0,read_bounds_from_delta:4,read_build:[0,2],read_rul:5,refer:0,run:0,simpl:[3,5,6,9,10,11],split:10,stage:11,submodul:[2,3,4,5,6,9,10,11],subpackag:[2,7],tabl:0,util:2,visual:3,welcom:0}})