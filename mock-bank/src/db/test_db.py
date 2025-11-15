# mock-bank/src/test_db.py
import asyncio
from session import AsyncSessionLocal
import crud

async def main():
    async with AsyncSessionLocal() as db:
        users = await crud.list_users(db, limit=10)
        print("Users:", len(users))
        for u in users:
            print(u.username, u.email)

        # sample: list first user's accounts + transactions
        if users:
            user = users[0]
            accounts = await crud.get_accounts_for_user(db, user.user_id)
            print("Accounts for", user.username, ":", len(accounts))
            if accounts:
                acct = accounts[0]
                txs = await crud.get_transactions_for_account(db, acct.account_id, limit=5)
                print("Recent txs:", len(txs))
                for t in txs:
                    print(t.transaction_reference, t.amount, t.status)

if __name__ == "__main__":
    asyncio.run(main())
