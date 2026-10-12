# Fraud Operations Policy (SAMPLE - fictional, for demo use)

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
