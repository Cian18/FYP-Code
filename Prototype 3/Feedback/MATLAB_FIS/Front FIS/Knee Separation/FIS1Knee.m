function fis = FIS1Knee(con_std, con_mean)
    % Initialize FIS
    fis = mamfis('Name', 'KneeSep');
    
    % Define input membership function parameters
    corr = [con_std, con_mean];
    med_neg = [con_std, con_mean - 4*con_std];
    med_pos = [con_std, con_mean + 4*con_std];
    lar_neg = [3*con_std, -2.5];
    lar_pos = [3*con_std, 2.5];
    
    % Add input with membership functions
    fis = addInput(fis, [-2.5, 2.5], 'Name', 'AngErr');
    fis = addMF(fis, 'AngErr', 'gaussmf', corr, 'Name', 'Correct');
    fis = addMF(fis, 'AngErr', 'gaussmf', med_neg, 'Name', 'Medium_neg');
    fis = addMF(fis, 'AngErr', 'gaussmf', med_pos, 'Name', 'Medium_pos');
    fis = addMF(fis, 'AngErr', 'gaussmf', lar_neg, 'Name', 'Large_neg');
    fis = addMF(fis, 'AngErr', 'gaussmf', lar_pos, 'Name', 'Large_pos');
    
    % Add output with membership functions
    fis = addOutput(fis, [-1.5, 1.5], 'Name', 'Output');
    fis = addMF(fis, 'Output', 'trapmf', [-1.5, -1.2, -0.8, -0.5], 'Name', 'Incorrect_Negative');
    fis = addMF(fis, 'Output', 'trapmf', [-0.8, -0.6, -0.4, -0.2], 'Name', 'Medium_Negative');
    fis = addMF(fis, 'Output', 'trapmf', [-0.5, -0.2, 0.2, 0.5], 'Name', 'Correct');
    fis = addMF(fis, 'Output', 'trapmf', [0.2, 0.4, 0.6, 0.8], 'Name', 'Medium_Positive');
    fis = addMF(fis, 'Output', 'trapmf', [0.5, 0.8, 1.2, 1.5], 'Name', 'Incorrect_Positive');
    

    %define rules
    fis = addRule(fis, 'If (AngErr is Correct) then (Output is Correct)');
    fis = addRule(fis, 'If (AngErr is Medium_neg) then (Output is Medium_Negative)');
    fis = addRule(fis, 'If (AngErr is Medium_pos) then (Output is Medium_Positive)');
    fis = addRule(fis, 'If (AngErr is Large_neg) then (Output is Incorrect_Negative)');
    fis = addRule(fis, 'If (AngErr is Large_pos) then (Output is Incorrect_Positive)');
end