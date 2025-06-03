function fis = FIS1TorsoSide(con_std, con_mean, instance_name)
    % Initialize FIS
    fis = mamfis('Name', instance_name);
    
    % Define input membership function parameters
    corr = [con_std, con_mean];
    lar_neg = [-0.04, -80];
    lar_pos = [0.025, 80];
    
    % Add input with membership functions
    fis = addInput(fis, [-180,180], 'Name', 'AngErr');
    fis = addMF(fis, 'AngErr', 'gaussmf', corr, 'Name', 'Correct');
    fis = addMF(fis, 'AngErr', 'sigmf', lar_neg, 'Name', 'Large_neg');
    fis = addMF(fis, 'AngErr', 'sigmf', lar_pos, 'Name', 'Large_pos');
    
    % Add output with membership functions
    fis = addOutput(fis, [-1.5, 1.5], 'Name', 'Output');
    fis = addMF(fis, 'Output', 'trapmf', [-1.5, -1.2, -0.8, -0.5], 'Name', 'Incorrect_Negative');
    fis = addMF(fis, 'Output', 'trapmf', [-0.5, -0.2, 0.2, 0.5], 'Name', 'Correct');
    fis = addMF(fis, 'Output', 'trapmf', [0.5, 0.8, 1.2, 1.5], 'Name', 'Incorrect_Positive');
    

    %define rules
    fis = addRule(fis, 'If (AngErr is Correct) then (Output is Correct)');
    fis = addRule(fis, 'If (AngErr is Large_neg) then (Output is Incorrect_Negative)');
    fis = addRule(fis, 'If (AngErr is Large_pos) then (Output is Incorrect_Positive)');
end