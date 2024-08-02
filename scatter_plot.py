import matplotlib.pyplot as plt
from statistics import mean, stdev
import argparse

fontsize_ticks = 14
fontsize_title = 20
fontsize_axis  = 18

palette = ['#f3a683',
           '#778beb',
           '#f19066',
           '#546de5',
           '#786fa6',
           '#ea8685']

# function to convert seconds in hour and minutes
def convert_sec(seconds): 
    hours         = seconds // 3600
    remaining_sec = seconds % 3600
    minutes       = remaining_sec // 60
    return hours, minutes

parser = argparse.ArgumentParser(description="Make scatter plots related to learning Hodgkin-Huxley model with DeepONet")
parser.add_argument("--arch", type=str, default="DON", help="DON, WNO or FNO")
args = parser.parse_args()

data_test = []
data_train = []
testlist = []
if args.arch=="DON":
    # test_name # width   # train                 # test                # Training time (s)
    test_105 =  [ 600,    [0.017887574955821037,
                           0.01992736928164959,
                           0.016779517754912376],[0.023890357613563538,
                                                  0.02582274079322815,
                                                  0.02307195544242859],  2140]
    test_106 =  [ 500,    [0.020711338222026823,
                           0.021566666439175607,
                           0.021887164413928985],[0.026597797274589538,
                                                  0.027479132413864137,
                                                  0.028008538484573364], 2122]
    test_107 =  [ 700,    [0.014863241501152515,
                           0.016362050995230675,
                           0.016580374389886857],[0.020823637843132018,
                                                  0.022337449193000795,
                                                  0.022792633175849914], 2149]
    test_108 =  [ 800,    [0.016361579969525336,
                           0.016996013671159743,
                           0.017238110303878784],[0.022679397463798524,
                                                  0.02313012719154358,
                                                  0.023540618419647216], 2146]
    test_109 =  [ 400,    [0.0215907521545887,
                           0.020713126510381697,
                           0.021509890481829642],[0.027747938632965086,
                                                  0.026663305163383486,
                                                  0.027671761512756347], 2114]
    test_110 =  [ 900,    [0.01650800310075283,
                           0.016369117647409438,
                           0.016699903681874274],[0.02320848822593689,
                                                  0.022828291356563567,
                                                  0.02325628459453583], 2212]
    test_111 =  [1000,    [0.01758544035255909,
                           0.01698608383536339,
                           0.017363536581397058],[0.02390597462654114,
                                                  0.02299903392791748,
                                                  0.023634833693504335], 2209]
    
    testlist =  [test_109,test_106,test_105,test_107,test_108,test_110,test_111] 
    #data_train = [(400, 0.0215), (500, 0.0207), (600, 0.0178), (700, 0.0148), (800, 0.0163), (900, 0.0165), (1000, 0.0175)]        
    #data_test = [(400, 0.0277), (500, 0.0265), (600, 0.0238), (700, 0.0208), (800, 0.0226), (900, 0.0232), (1000, 0.0239)]
elif args.arch=="FNO":
    test_102 = [16 ,[0.008997942544519901,
                     0.013172060064971448,
                     0.012949854545295238],[0.011693967580795288,
                                            0.016057044565677643,
                                            0.015100511908531188], 699]
    test_103 = [8  ,[0.013606304600834847,
                     0.015975375846028327,
                     0.01646119274199009],[0.01590629756450653,
                                           0.01889709144830704,
                                           0.019059271216392518], 699]
    test_104 = [32 ,[0.007923789527267218,
                     0.009242457486689091,
                     0.016180948950350285],[0.012372631132602691,
                                            0.012897460162639618,
                                            0.019451242685317994], 699]
    test_105 = [64 ,[0.015071362927556039,
                     0.013226330801844597,
                     0.028409363925457],[0.02076708376407623,
                                         0.017779705226421357,
                                         0.03196794629096985], 699]
    test_106 = [128,[0.016225730292499064,
                     0.013659143634140492,
                     0.016005187779664993],[0.022039121985435485,
                                            0.018481670320034026,
                                            0.022810511589050293], 702]
    testlist = [test_103,test_102,test_104,test_105,test_106]
    #data_train = [(8, 0.0136), (16, 0.0089), (32, 0.0079), (64, 0.01507), (128, 0.0162)]
    #data_test = [(8, 0.0159), (16, 0.01169), (32, 0.0123), (64, 0.0207), (128, 0.02203)]
elif args.arch=="WNO":
    test_030 = [256,[0.0033369118813425303,
                     0.004240718241780996,
                     0.003245361070148647],[0.03339563310146332,
                                            0.03666243016719818,
                                            0.033568723797798156],15559]
    test_031 = [8,  [0.020278269052505495,
                     0.03607741504907608,
                     0.020708807334303855],[0.049424048662185666,
                                            0.05896107912063599,
                                            0.04298272609710693],4059]
    test_032 = [16, [0.01595795176923275,
                     0.01395194634795189,
                     0.015198439247906209],[0.045368095636367796,
                                            0.03729689836502075,
                                            0.04135965228080749],4061]
    test_033 = [32, [0.012262163087725639,
                     0.010551054924726487,
                     0.012339884974062442],[0.0419635272026062,
                                            0.03846832811832428,
                                            0.04039208769798279],4209]
    test_034 = [64, [0.0077785498276352885,
                     0.007049382459372282,
                     0.006101500261574983],[0.03256745159626007,
                                            0.032943731546401976,
                                            0.03355768084526062],4519]
    test_035 = [128,[0.005386492200195789,
                     0.00501396139152348,
                     0.0043311968259513375],[0.03500817656517029,
                                             0.035144179463386535,
                                             0.03334957957267761],4736]
    testlist = [test_031,test_032,test_033,test_034,test_035,test_030]
    #data_train = [(8, 0.02027), (16, 0.01595), (32, 0.01226), (64, 0.00777), (128, 0.00538), (256, 0.00333)]
    #data_test  = [(8, 0.0494), (16, 0.0453), (32, 0.04196), (64, 0.03256), (128, 0.035008), (256, 0.03339)]
elif args.arch=="data_eff":
    don_9010 = ['90-10',[0.01152,
                         0.01216,
                         0.01233],[0.02461,
                                   0.02313,
                                   0.02844]]
    
    don_8020 = ['80-20',[0.01486,
                         0.01636,
                         0.01658],[0.02082,
                                   0.02234,
                                   0.02279]]
    don_7030 = ['70-30',[0.01517,
                         0.01355,
                         0.01056],[0.03171,  
                                   0.02838, 
                                   0.02614]]
    don_6040 = ['60-40',[0.01085,
                         0.01255,
                         0.01531],[0.02796, 
                                   0.02843,
                                   0.03090]]
    don_5050 = ['50-50',[0.01180,
                         0.01807,
                         0.01724],[0.03193, 
                                   0.03412, 
                                   0.03328]]
    
    fno_9010 = ['90-10',[0.01405, 
                         0.01417, 
                         0.01298],[0.01660, 
                                   0.01801, 
                                   0.01439]]
    fno_8020 = ['80-20',[0.00900,
                         0.01317,
                         0.01295],[0.01169,
                                   0.01606,
                                   0.01510]]
    fno_7030 = ['70-30',[0.01156,
                         0.01151,
                         0.01006],[0.01246,    
                                   0.02214, 
                                   0.01244]]
    fno_6040 = ['60-40',[0.01336,
                         0.01294,
                         0.01281],[0.01559,
                                   0.01589,
                                   0.01518]]
    fno_5050 = ['50-50',[0.01550,
                         0.01843,
                         0.02307],[0.02041,
                                   0.02561,
                                   0.01931]]
    
    wno_9010 = ['90-10',[0.01055,
                         0.01031,
                         0.00840],[0.03263, 
                                   0.03603,
                                   0.03356]]
    wno_8020 = ['80-20',[0.00778,
                         0.00705,
                         0.00610],[0.03257,
                                   0.03294,
                                   0.03356]]
    wno_7030 = ['70-30',[0.01508,
                         0.00928,
                         0.01060],[0.04045,  
                                   0.03930, 
                                   0.03998]]
    wno_6040 = ['60-40',[0.01094,
                         0.02611,
                         0.01031],[0.04125,  
                                   0.05399, 
                                   0.03736]]
    wno_5050 = ['50-50',[0.01144,
                         0.01647,
                         0.01019],[0.04271, 
                                   0.05136,
                                   0.04152]]
    
    testlist = [don_9010, don_8020, don_7030, don_6040, don_5050,
                fno_9010, fno_8020, fno_7030, fno_6040, fno_5050,
                wno_9010, wno_8020, wno_7030, wno_6040, wno_5050]
else:
    ValueError("Arch type not DON, FNO, WNO or data_eff")

for test in testlist:
        data_train.append(( test[0],mean(test[1]),stdev(test[1]) ))
        data_test.append((  test[0],mean(test[2]),stdev(test[2]) ))
# Print values for latex table
for test in data_test:
    print("{:.4f}$\pm${:.4f} & [{}]*3 & [{}]*4 ".format(test[1],test[2],test[0],test[0]))

# Print training times
#for test in testlist:
#    hrs, mins = convert_sec(test[3])
#    print("{}h {}min".format(hrs,mins))
# Extract x and y values from the list of tuples
x_values_train = [x for x,_,_ in data_train]
y_values_train = [y for _,y,_ in data_train]
y_err_train    = [e for _,_,e in data_train]

x_values_test = [x for x,_,_ in data_test]
y_values_test = [y for _,y,_ in data_test]
y_err_test    = [e for _,_,e in data_test]

# Create a scatter plot
if args.arch=="data_eff":
    fig, axs = plt.subplots(1, 3, figsize=(15,6))
    # don
    axs[0].errorbar(x_values_train[0:5], y_values_train[0:5], yerr=y_err_train[0:5], label='Train DON', c=palette[1], marker="o",capsize=4, markersize=7, linewidth=3)
    axs[0].errorbar(x_values_test[0:5], y_values_test[0:5], yerr=y_err_test[0:5], label='Test DON', c=palette[0], marker="o",capsize=4, markersize=7, linewidth=3)
    # fno
    axs[1].errorbar(x_values_train[5:10], y_values_train[5:10], yerr=y_err_train[5:10], label='Train FNO', c=palette[3], marker="o",capsize=4, markersize=7, linewidth=3)
    axs[1].errorbar(x_values_test[5:10], y_values_test[5:10], yerr=y_err_test[5:10], label='Test FNO', c=palette[2], marker="o",capsize=4, markersize=7, linewidth=3)
    # wno
    axs[2].errorbar(x_values_train[10:], y_values_train[10:], yerr=y_err_train[10:], label='Train WNO', c=palette[5], marker="o",capsize=4, markersize=7, linewidth=3)
    axs[2].errorbar(x_values_test[10:], y_values_test[10:], yerr=y_err_test[10:], label='Test WNO', c=palette[4], marker="o",capsize=4, markersize=7, linewidth=3)
else:
    plt.figure(figsize=(9.5,6)) 
    plt.errorbar(x_values_train, y_values_train, yerr=y_err_train, label='Train error', marker="o",capsize=4, markersize=7, linewidth=3)
    plt.errorbar(x_values_test, y_values_test, yerr=y_err_test, label='Test error', marker="o",capsize=4, markersize=7, linewidth=3)

# Add labels and title
if args.arch=="DON": 
    plt.xlabel('Network width',fontsize=fontsize_axis)
    plt.title('$L^2$ Relative error vs Network Width',fontsize=fontsize_title)
    plt.ylabel('$L^2$ relative error',fontsize=fontsize_axis)
    plt.xticks(x_values_test,fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize_axis)
    plt.grid()
elif args.arch=="FNO":
    plt.xlabel('Fourier Modes',fontsize=fontsize_axis)
    plt.title('$L^2$ Relative error vs Fourier Modes',fontsize=fontsize_title)
    plt.ylabel('$L^2$ relative error',fontsize=fontsize_axis)
    plt.xticks(x_values_test,fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize_axis)
    plt.grid()
elif args.arch=="WNO":
    plt.xlabel('$R_{\\theta_t}$ width',fontsize=fontsize_axis)
    plt.title('$L^2$ Relative Error vs $R_{\\theta_t}$ width',fontsize=fontsize_title)
    plt.ylabel('$L^2$ relative error',fontsize=fontsize_axis)
    plt.xticks(x_values_test,fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize_axis)
    plt.grid()
elif args.arch=="data_eff":
    #axs[0].xlabel(,fontsize=fontsize_axis)
    fig.suptitle('Data Efficiency Test',fontsize=fontsize_title)
    # Labels for each subplot (these are the same for each, but can be adjusted as needed)
    xlabel = 'Training Data - Test Data (%)'
    ylabel = '$L^2$ relative error'

    # Plot data and set properties for each subplot
    axs[0].set_title('DON', fontsize=fontsize_title)
    axs[0].set_ylabel(ylabel, fontsize=fontsize_axis)
    axs[0].set_xticks(x_values_test)
    axs[0].tick_params(axis='x', labelsize=fontsize_ticks)
    axs[0].tick_params(axis='y', labelsize=fontsize_ticks)
    axs[0].legend(fontsize=fontsize_axis)
    axs[0].grid(True)

    axs[1].set_title('FNO', fontsize=fontsize_title)
    axs[1].set_xlabel(xlabel, fontsize=fontsize_axis)
    axs[1].set_xticks(x_values_test)
    axs[1].tick_params(axis='x', labelsize=fontsize_ticks)
    axs[1].tick_params(axis='y', labelsize=fontsize_ticks)
    axs[1].legend(fontsize=fontsize_axis)
    axs[1].grid(True)

    axs[2].set_title('WNO', fontsize=fontsize_title)
    axs[2].set_xticks(x_values_test)
    axs[2].tick_params(axis='x', labelsize=fontsize_ticks)
    axs[2].tick_params(axis='y', labelsize=fontsize_ticks)
    axs[2].legend(fontsize=fontsize_axis)
    axs[2].grid(True)
    plt.tight_layout()
    plt.savefig("data_eff.eps", format='eps')
else:
    ValueError("Arch type not DON, FNO or WNO")

# Show plot
plt.show()