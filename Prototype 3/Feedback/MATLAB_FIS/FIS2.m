function fis2 = FIS2(instance_name)
fis2 = mamfis('Name', instance_name);


%add inputs and create membership functions
fis2 = addInput(fis2, [-1,1], 'Name', "A");
fis2 = addMF(fis2, 'A', 'gaussmf', [0.23, 0], 'Name', 'Correct');
fis2 = addMF(fis2, 'A', 'gaussmf', [0.16, -0.5], 'Name', 'Medium_neg');
fis2 = addMF(fis2, 'A', 'gaussmf', [0.16, 0.5], 'Name', 'Medium_pos');
fis2 = addMF(fis2, 'A', 'gaussmf', [0.25, -1], 'Name', 'Incorrect_neg');
fis2 = addMF(fis2, 'A', 'gaussmf', [0.25, 1], 'Name', 'Incorrect_pos');

fis2 = addInput(fis2, [-1,1], 'Name', "B");
fis2 = addMF(fis2, 'B', 'gaussmf', [0.23, 0], 'Name', 'Correct');
fis2 = addMF(fis2, 'B', 'gaussmf', [0.16, -0.5], 'Name', 'Medium_neg');
fis2 = addMF(fis2, 'B', 'gaussmf', [0.16, 0.5], 'Name', 'Medium_pos');
fis2 = addMF(fis2, 'B', 'gaussmf', [0.25, -1], 'Name', 'Incorrect_neg');
fis2 = addMF(fis2, 'B', 'gaussmf', [0.25, 1], 'Name', 'Incorrect_pos');

%Add output and create membership functions
fis2 = addOutput(fis2, [-2,2], 'Name', 'C');
fis2 = addMF(fis2, 'C', 'gaussmf', [0.23, 0], 'Name', 'Correct');
fis2 = addMF(fis2, 'C', 'gaussmf', [0.16, -0.5], 'Name', 'Medium_neg');
fis2 = addMF(fis2, 'C', 'gaussmf', [0.16, 0.5], 'Name', 'Medium_pos');
fis2 = addMF(fis2, 'C', 'gaussmf', [0.25, -1], 'Name', 'Incorrect_neg');
fis2 = addMF(fis2, 'C', 'gaussmf', [0.25, 1], 'Name', 'Incorrect_pos');

%Define rules
%correct output
fis2 = addRule(fis2, 'If (A is Correct) and (B is Correct) then (C is Correct)');

%medium negative output
fis2 = addRule(fis2, 'If (A is Medium_neg) and (B is Medium_neg) then (C is Medium_neg)');
fis2 = addRule(fis2, 'If (A is Medium_neg) and (B is Correct) then (C is Medium_neg)');
fis2 = addRule(fis2, 'If (A is Correct) and (B is Medium_neg) then (C is Medium_neg)');
%prioritization of conflict for medium negative
fis2 = addRule(fis2, 'If (A is Medium_pos) and (B is Medium_neg) then (C is Medium_neg)');
fis2 = addRule(fis2, 'If (A is Medium_neg) and (B is Medium_pos) then (C is Medium_neg)');

%medium positive output
fis2 = addRule(fis2, 'If (A is Medium_pos) and (B is Medium_pos) then (C is Medium_pos)');
fis2 = addRule(fis2, 'If (A is Medium_pos) and (B is Correct) then (C is Medium_pos)');
fis2 = addRule(fis2, 'If (A is Correct) and (B is Medium_pos) then (C is Medium_pos)');

%incorrect negative output
fis2 = addRule(fis2, 'If (A is Incorrect_neg) and (B is Correct) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Correct) and (B is Incorrect_neg) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Incorrect_neg) and (B is Medium_neg) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Medium_neg) and (B is Incorrect_neg) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Incorrect_neg) and (B is Incorrect_neg) then (C is Incorrect_neg)');
%prioritization of conflict for incorrect negative
fis2 = addRule(fis2, 'If (A is Incorrect_neg) and (B is Medium_pos) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Medium_pos) and (B is Incorrect_neg) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Incorrect_pos) and (B is Incorrect_neg) then (C is Incorrect_neg)');
fis2 = addRule(fis2, 'If (A is Incorrect_neg) and (B is Incorrect_pos) then (C is Incorrect_neg)');

%incorrect positive output
fis2 = addRule(fis2, 'If (A is Incorrect_pos) and (B is Correct) then (C is Incorrect_pos)');
fis2 = addRule(fis2, 'If (A is Correct) and (B is Incorrect_pos) then (C is Incorrect_pos)');
fis2 = addRule(fis2, 'If (A is Incorrect_pos) and (B is Medium_pos) then (C is Incorrect_pos)');
fis2 = addRule(fis2, 'If (A is Medium_pos) and (B is Incorrect_pos) then (C is Incorrect_pos)');
fis2 = addRule(fis2, 'If (A is Incorrect_pos) and (B is Incorrect_pos) then (C is Incorrect_pos)');
%prioritization of conflict for incorrect negative
fis2 = addRule(fis2, 'If (A is Incorrect_pos) and (B is Medium_neg) then (C is Incorrect_pos)');
fis2 = addRule(fis2, 'If (A is Medium_neg) and (B is Incorrect_pos) then (C is Incorrect_pos)');


end