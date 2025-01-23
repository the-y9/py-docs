#%%
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
#%%
ar = [1,2,3,3,3,2,1,2,3,4,1]
head = ListNode(ar[0])
current = head
for val in ar[1:]:
    current.next = ListNode(val)
    current = current.next
#%%
current = head
while current:
    print(current.val, end='->' if current.next else "")
    current = current.next
#%%
# Definition for singly-linked list.

def sortList(head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        
def split(head):
            slow = fast = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            mid = slow.next
            print(head.val,slow.val,mid.val,fast.val)
            slow.next = None
            return head, mid
        
def merge(left, right):
            dummy = ListNode()
            current = dummy
            while left and right:
                if left.val < right.val:
                    current.next = left
                    left = left.next
                else:
                    current.next = right
                    right = right.next
                current = current.next
            current.next = left if left else right
            return dummy.next
        
left, right = split(head)
left = sortList(left)
right = sortList(right)

# print(merge(left, right))


# %%
