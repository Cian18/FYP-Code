#This function extracts the features of a FIS exported from matlab. These will later be used to reconstruct the FIS.
def read_MATFIS(path):
    import scipy.io
    
    #load the FIS
    data = scipy.io.loadmat(path)
    
    #extract each FIS - there is a lot of indexing because the matlab format is very weird
    fis11 = data['u'][0,0]
    fis21 = data['u1'][0,0]
    fis22 = data['u2'][0,0]
    
    #Extract the inputs, ouputs and rules for each FIS    
    inp = [fis11[7][0],fis21[7][0],fis22[7][0]]
    outp = [fis11[8][0],fis21[8][0],fis22[8][0]]
    rules = [fis11[9][0],fis21[9][0],fis22[9][0]]
    
    #Extract the ranges for each of the inputs and outputs
#     in_range = [inp[0][2][0], inp[1][2][0], inp[2][2][0]]
#     out_range = [outp[0][2], outp[1][2], outp[2][2]]
    
    
    #Extract the input and output membership functions of each FIS
    in_mem = [[],[],[]]
    in_range = [[], [], []]
    #loop over each FIS
    for k in range(len(inp)):        
        #loop over all inputs for the current FIS
        for j in range(len(inp[k])):
            #define the current input for this FIS
            c_inp = inp[k][j]
            #initiate a dictionary that will hold the membership functions for the current input
            c_mem = {}
            #fill in the in_range list that will contain the range of each input
            in_range[k].append(c_inp[2][0])
            #loop over each membership function in the current input for the current FIS
            for i in range(c_inp[3].shape[1]):
                if j == 0:
                    c_mem[c_inp[3][0][i][0][0]] = (c_inp[3][0][i][2][0], c_inp[3][0][i][4][0])
                elif j == 1:
                    c_mem[c_inp[3][0][i][0][0]] = (c_inp[3][0][i][2][0], c_inp[3][0][i][4][0])
                else:
                    c_mem[c_inp[3][0][i][0][0]] = (c_inp[3][0][i][2][0], c_inp[3][0][i][4][0])
        
            #Append the MFs for the current input          
            in_mem[k].append(c_mem)
        
                
    out_mem = [[], [], []]
    out_range = [[], [], []]
    #loop over each FIS
    for k in range(len(outp)):        
        #loop over all inputs for the current FIS
        for j in range(len(outp[k])):
            #define the current input for this FIS
            c_outp = outp[k][j]
            #initiate a dictionary that will hold the membership functions for the current input
            c_mem = {}
            #fill in the in_range list that will contain the range of each input
            out_range[k].append(c_outp[2][0])
            #loop over each membership function in the current input for the current FIS
            for i in range(c_outp[3].shape[1]):
                if j == 0:
                    c_mem[c_outp[3][0][i][0][0]] = (c_outp[3][0][i][2][0], c_outp[3][0][i][4][0])
                elif j == 1:
                    c_mem[c_outp[3][0][i][0][0]] = (c_outp[3][0][i][2][0], c_outp[3][0][i][4][0])
                else:
                    c_mem[c_outp[3][0][i][0][0]] = (c_outp[3][0][i][2][0], c_outp[3][0][i][4][0])
        
            #Append the MFs for the current input          
            out_mem[k].append(c_mem)
                
    #Extract the rules for each FIS
    rul = [[], [], []]
    for j in range(len(rules)):
        for i in range(len(rules[j])):
            if j == 0:
                rul[j].append([rules[0][i][0][0], rules[0][i][1][0]])
            elif j == 1:
                rul[j].append([rules[1][i][0][0], rules[1][i][1][0]])
            else:
                rul[j].append([rules[2][i][0][0], rules[2][i][1][0]])
    
    #this function returns three lists, input and output membership function and the rules for each of the three FIS
    return in_range, in_mem, out_range, out_mem, rul








#define a function to re-construct 3 FIS exported from matlab. There are three FIS exported that form a FIS tree.
def re_construct(path):
    import numpy as np
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    
    in_range, in_mem, out_range, out_mem, rul = read_MATFIS(path)
    
    #define a list that will hold a dictionary to define each FIS input
    FIS_in = [{}, {}, {}]
    
    #define all the input membership functions
    #loop over each FIS
    for i in range(len(in_mem)):
        #loop over each input for the current FIS
        for k in range(len(in_mem[i])):
            #define the current input for the current FIS
            fis_range = in_range[i][k]
            FIS_in[i][f'fisin{k}'] = ctrl.Antecedent(np.linspace(fis_range[0], fis_range[1],500),f'Input{k}')

            #extract the name of the mombership functions
            keys = in_mem[i][k].keys()
            for j, key in enumerate(keys):
                #extract the type of membership function
                typ = in_mem[i][k][key][0]

                #extract the MF numerical definition
                mf = in_mem[i][k][key][1]

                if typ == 'gaussmf':
                    FIS_in[i][f'fisin{k}'][key] = fuzz.gaussmf(FIS_in[i][f'fisin{k}'].universe, mf[1], mf[0])
                elif typ == 'trapmf':
                    FIS_in[i][f'fisin{k}'][key] = fuzz.trapmf(FIS_in[i][f'fisin{k}'].universe, mf)
                elif typ == 'sigmf':
                    FIS_in[i][f'fisin{k}'][key] = fuzz.sigmf(FIS_in[i][f'fisin{k}'].universe, mf[1], mf[0])
                else:
                    return 'unknown membership function type'

    #define a list that will hold a dictionary to define each FIS input
    FIS_out = [{}, {}, {}]
    
    #define all the input membership functions
    #loop over each FIS
    for i in range(len(out_mem)):
        #loop over each input for the current FIS
        for k in range(len(out_mem[i])):
            #define the current input for the current FIS
            fis_range = out_range[i][k]
            FIS_out[i][f'fisout{k}'] = ctrl.Consequent(np.linspace(fis_range[0],fis_range[1],500),f'Output{k}')

            #extract the name of the mombership functions
            keys = out_mem[i][k].keys()
            for j, key in enumerate(keys):
                #extract the type of membership function
                typ = out_mem[i][k][key][0]

                #extract the MF numerical definition
                mf = out_mem[i][k][key][1]
                
                if typ == 'gaussmf':
                    FIS_out[i][f'fisout{k}'][key] = fuzz.gaussmf(FIS_out[i][f'fisout{k}'].universe, mf[1], mf[0])
                elif typ == 'trapmf':
                    FIS_out[i][f'fisout{k}'][key] = fuzz.trapmf(FIS_out[i][f'fisout{k}'].universe, mf)
                elif typ == 'sigmf':
                    FIS_out[i][f'fisout{k}'][key] = fuzz.sigmf(FIS_out[i][f'fisout{k}'].universe, mf[1], mf[0])
                else:
                    return 'unknown membership function type'
                
    #Define the rules for each FIS as a list containing a list of rules for each FIS
    rules = [[], [], []]
    
    #loop over each FIS
    for i in range(len(rul)):
        #loop over each rule
        for j in range(len(rul[i])):
            #I know that the first FIS has 1 input and one output and that the other two have 2 inputs and one output.
            #Doing this step based on this knowledge makes it less generalizable, but it is going to be faster to code
            if i == 0:
                #extract the keys of the input and output membership functions
                in_keys = list(in_mem[i][0].keys())
                out_keys = list(out_mem[i][0].keys())
                
                #define the index of the current rule (i.e. which membership functions it is referring to)
                in_ind = rul[i][j][0][0]-1
                out_ind = in_ind = rul[i][j][1][0]-1

                #define the rule
                rules[i].append(ctrl.Rule(FIS_in[i]['fisin0'][in_keys[in_ind]], FIS_out[i]['fisout0'][out_keys[out_ind]]))
                
            elif i == 1 or i == 2:
                #extract the keys of the input and output membership functions
                in1_keys = list(in_mem[i][0].keys())
                in2_keys = list(in_mem[i][1].keys())
                out_keys = list(out_mem[i][0].keys())
                
                #define the index of the current rule (i.e. which membership functions it is referring to)
                in1_ind = rul[i][j][0][0]-1
                in2_ind = rul[i][j][0][1]-1
                out_ind = rul[i][j][1][0]-1
                
                #define the rule
                rules[i].append(ctrl.Rule(FIS_in[i]['fisin0'][in1_keys[in1_ind]] & FIS_in[i]['fisin1'][in2_keys[in2_ind]], FIS_out[i]['fisout0'][out_keys[out_ind]]))

    #Finally, create each FIS based on the rules
    FIS11 = ctrl.ControlSystem(rules[0])
    FIS21 = ctrl.ControlSystem(rules[1])
    FIS22 = ctrl.ControlSystem(rules[2])

    return FIS11, FIS21, FIS22









