%get first N existing prime numbers
function p_list = get_nPrime(n)
    i = n;
    p_list = primes(i);
    while length(p_list) < n
        i = i+n;
        p_list = primes(i);
    end
end