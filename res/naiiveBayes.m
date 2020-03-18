function [classes] = naiiveBayes(A, R, T)
% Define V(ocab), C(lasses)
V = [];
C = [ "Accept"; "Review" ];

% Calculate probability of each class
PA = length(A) / ( length(A) + length(R) );
PR = length(R) / ( length(A) + length(R) );

% Define empty word vectors
AW = [];
RW = [];

% Go through A vector to create a word vocabulary
for i = 1:length(A)
    words = split(A(i), " ");
    for j = 1:length(words)
        try
            validatestring(words(j), V);
        catch
            % Add word to vocabulary if it doesn't exist
            V = [ V, words(j) ];
        end

        % Always add to A vocabulary
        AW = [ AW, words(j) ];
    end
end

% Convert A vocabulary into an nx2 matrix of word -> count pairs
[ nr, ~, rc ] = unique(AW);
nrc = histcounts(rc, length(nr));
AW = [ nr(:), nrc(:) ];

% Go through R vector to create a word vocabulary
for i = 1:length(R)
    words = split(R(i), " ");
    for j = 1:length(words)
        try
            validatestring(words(j), V);
        catch
            % Add word to vocabulary if it doesn't exist
            V = [ V, words(j) ];
        end

        % Always add to R vocabulary
        RW = [ RW, words(j) ];
    end
end

% Convert R vocabulary into an nx2 matrix of word -> count pairs
[ nr, ~, rc ] = unique(RW);
nrc = histcounts(rc, length(nr));
RW = [ nr(:), nrc(:) ];

% Define an unclassified string vector
classes = [ "unclassified" ];

% Go through Testing list to classify words
for i = 1:length(T)
    % Define word vectors, length of vocabularies and initialise Testing
    % Prediction vectors
    TW = split(T(i), " ");
    Nv = length(V);
    Na = length(AW);
    Nr = length(RW);
    TA = 0;
    TR = 0;

    % Create probability for each word
    for j = 1:length(TW)
        % Check if word is in known vocabulary
        [ f, n ] = ismember( TW(j), AW );

        % If word is known, find it's how common it is otherwise assume 0
        if f
            count = str2num( AW(n,2) );
        else
            count = 0;
        end

        % Calculate probabilities of word given class, and then apply
        % Bayes' Theorem
        PWA = ( count + 1 ) / ( Na + Nv );
        TA = TA + log10( ( PWA * PA ) / ( ( PWA * PA ) + ( ( 1 - PWA ) * ( 1 - PA ) ) ) );

        % Check if word is in known vocabulary
        [ f, n ] = ismember( TW(j), RW );

        % If word is known, find it's how common it is otherwise assume 0
        if f
            count = str2num( RW(n,2) );
        else
            count = 0;
        end

        % Calculate probabilities of word given class, and then apply
        % Bayes' Theorem
        PWR = ( count + 1 ) / ( Nr + Nv );
        TR = TR + log10( ( PWR * PR ) / ( ( PWR * PR ) + ( ( 1 - PWR ) * ( 1 - PR ) ) ) );
    end

    % Evaluate which class has the strong association and add
    % classification to vector
    if TA > TR
        classes(i) = C(1);
    else
        classes(i) = C(2);
    end
end
end
