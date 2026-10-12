from pathlib import Path

Path("docs").mkdir(exist_ok=True)

POLICY = """# Fraud Operations Policy (SAMPLE - fictional, for demo use)

## 1. Velocity rules
Three or more transactions on the same card within one hour is a velocity alert. Five or more transactions within one hour is treated as card testing unless the cardholder is reachable and confirms the activity. Velocity alerts must be reviewed within 15 minutes.

## 2. High-value transactions
A transaction above 500 USD that is more than five times the card's 30-day average spend requires manual review before settlement. Any transaction above 1,000 USD is escalated to a senior analyst regardless of the model score.

## 3. Foreign transactions
A foreign transaction on a card with no foreign activity in the last 90 days raises the risk level by one step. A foreign transaction between midnight and 5 am local time raises it by a second step.

## 4. Night-time activity
Activity between midnight and 5 am is higher risk. Night-time activity combined with high value or high velocity moves the case to priority review.

## 5. Escalation and response times
Priority cases must be handled within 15 minutes and standard cases within 4 hours. Blocking a card with more than 3 confirmed fraudulent transactions needs senior analyst sign-off. Cases with total exposure above 5,000 USD go to the fraud manager.

## 6. Customer contact
Contact the cardholder only through the registered phone number or app notification. Never ask for the full card number, PIN, or one-time passcode. If the cardholder cannot be reached after two attempts within 30 minutes, apply a temporary hold on the card.

## 7. False positives
If the cardholder confirms the transaction, release the hold within 5 minutes, record the reason, and add the merchant to the cardholder's trusted list. Three false positives on the same card within 30 days trigger a model feedback review.

## 8. Documentation
Every case must record the transaction ids, model score, top explanation factors, the policy rule matched, the analyst decision, and the outcome. Case records are kept for 7 years.
"""

CASES = """# Past Case Notes (SAMPLE - fictional, for demo use)

## Case 2025-014: Gas station burst
A card was used 6 times in 8 minutes at different merchants, with amounts between 120 and 480 USD. This was card testing followed by high-value purchases. Outcome: confirmed fraud, card blocked after the third alert, total loss 1,850 USD. Lesson: five or more transactions in under 10 minutes is almost always fraud.

## Case 2025-021: Foreign late-night purchase
A single 640 USD foreign transaction at 3 am on a card with no foreign history. The cardholder confirmed they were travelling, so this was a false positive. Lesson: check recent travel notices and app location before blocking.

## Case 2025-033: Small tests then a large purchase
Two 1 USD transactions were followed by a 900 USD electronics purchase within 20 minutes of a password reset. Outcome: confirmed account takeover fraud. Lesson: tiny test transactions followed by a large amount is a strong fraud pattern.

## Case 2025-038: Holiday shopping false alarm
Four transactions in one hour at a shopping mall, 380 USD in total, all domestic and in the daytime, with amounts similar to the cardholder's history. Outcome: false positive. Lesson: velocity alone, with normal amounts in the daytime, is low risk.

## Case 2025-045: Dormant card suddenly spends
A card that averaged 25 USD a month made a 1,200 USD online purchase with a foreign merchant. Outcome: confirmed fraud. Lesson: an amount more than 20 times the card's average is a very strong signal.

## Case 2025-052: Recurring subscription flagged
A recurring 49 USD subscription was flagged three months in a row. The merchant was added to the trusted list and the case was sent for model feedback review. Lesson: recurring identical amounts from the same merchant are usually safe.
"""

Path("docs/fraud_policy.md").write_text(POLICY, encoding="utf-8")
Path("docs/case_notes.md").write_text(CASES, encoding="utf-8")
print("Created docs/fraud_policy.md and docs/case_notes.md")