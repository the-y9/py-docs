import sys
from typing import List

'''
The script returns the sum of the fourth powers of all non-positive integers using recursion.
Uses standard input and output.
Returns -1 number of integers does not match X.
No blank lines are present in start, end or between test cases.
Additional:
- If any non-integer is fed as input, it returns -1 for that test case.
- If X empty, it returns -1.
- If number of test cases is is non-positive or non-numeric, it returns -1.
'''

def power_yn(X: int, Yn: List[str], current_index: int, case_sum: int) -> int:
    if current_index >=X:
        return case_sum
    try:
        y = int(Yn[current_index])
    except (ValueError, IndexError) as e:
        return -1
    if y<=0:
        new_sum = case_sum + (y**4)
    else:
        new_sum = case_sum
    return power_yn(X, Yn, current_index + 1, new_sum)

def test_cases(n):
    if n<=0:
        return []
    try:
        X = int(sys.stdin.readline())
    except ValueError:
        return ["-1"] + test_cases(n-1)
    Yn = sys.stdin.readline().split()
    if X != len(Yn):
        return ["-1"] + test_cases(n-1)

    return [str(power_yn(X,Yn,0,0))] + test_cases(n-1)

def main():
    try:
        n = int(sys.stdin.readline())
    except ValueError:
        sys.stderr.write("-1")
        return
    if n <= 0:
        sys.stdout.write("-1")
        return
    
    solutions = test_cases(n)
    sys.stdout.write('\n'.join(solutions))

if __name__ == '__main__':
    main()

#%%

import requests, json, time, hmac, hashlib, base64, struct

def generate_totp_sha512(secret, interval=30, digits=10):
    key = secret.encode('ascii')
    counter = int(time.time() // interval)
    msg = struct.pack(">Q", counter)
    hmac_digest = hmac.new(key, msg, hashlib.sha512).digest()
    offset = hmac_digest[-1] & 0x0F
    truncated_hash = hmac_digest[offset:offset+4]
    code = struct.unpack(">I", truncated_hash)[0]
    code &= 0x7FFFFFFF
    token = code % (10 ** digits)
    return str(token).zfill(digits)

email = "yashofc9@gmail.com"
secret = email + "HENNGECHALLENGE004"
print("Secret:", secret)
totp = generate_totp_sha512(secret)
print("TOTP:", totp)


auth_string = f"{email}:{totp}"
auth_header = "Basic " + base64.b64encode(auth_string.encode()).decode()
print(auth_header)


url = "https://api.challenge.hennge.com/challenges/backend-recursion/004"

data = {
  "github_url": "https://gist.github.com/the-y9/20c8fd6ab1fffb7c8a2fd29876c4a07e",
  "contact_email": "yashofc9@gmail.com",
  "solution_language": "python"
}

headers = {
    "Content-Type": "application/json",
    "Authorization": auth_header
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.status_code)
print(response.text)

"""
Secret: yashofc9@gmail.comHENNGECHALLENGE004
TOTP: 1137071049
Basic eWFzaG9mYzlAZ21haWwuY29tOjExMzcwNzEwNDk=      
200
{"message":"Congratulations! You have achieved mission 3"}                    

"""