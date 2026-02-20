# User Story and Acceptance Criteria Template

A user story captures a software feature from an end-user perspective.
Acceptance criteria define the conditions that must be met for the story to be "done."

---

## User Story Format

```
Title: [Short descriptive title]

As a [role/persona],
I want [feature/capability],
So that [benefit/value].

Priority: [Must Have / Should Have / Could Have / Won't Have]
Story Points: [1 / 2 / 3 / 5 / 8 / 13]
Sprint: [Sprint number or Backlog]
```

## Acceptance Criteria Format (Gherkin / Given-When-Then)

```
Scenario: [Scenario title]
  Given [initial context / precondition]
  When  [action performed by user or system]
  Then  [expected outcome]
  And   [additional outcome] (optional)
```

---

## Example 1: Checkout Flow

**Title:** Complete purchase with saved payment method

As a returning customer,
I want to check out using my saved credit card,
So that I can complete my purchase quickly without re-entering payment details.

**Priority:** Must Have
**Story Points:** 5
**Sprint:** Sprint 3

### Acceptance Criteria

**Scenario 1: Successful checkout with saved card**
```
Given I am logged in and have items in my cart
  And I have at least one saved payment method on file
When I proceed to checkout and select my saved card
  And I confirm the order
Then the order is placed successfully
  And I receive an order confirmation email within 2 minutes
  And my cart is cleared
```

**Scenario 2: Saved card is expired**
```
Given I am logged in and have items in my cart
  And my only saved card has expired
When I proceed to checkout
Then I am prompted to add a new payment method
  And my saved card is flagged as expired in my account settings
```

**Scenario 3: Payment is declined**
```
Given I am logged in and attempt to place an order
When the payment gateway declines the transaction
Then the order is not created
  And I am shown a clear error message with next steps
  And my cart contents are preserved
```

---

## Example 2: Product Search

**Title:** Search products by keyword with filters

As a shopper,
I want to search for products by keyword and filter results by category and price,
So that I can quickly find items that match my needs.

**Priority:** Must Have
**Story Points:** 8
**Sprint:** Sprint 2

### Acceptance Criteria

**Scenario 1: Keyword search returns relevant results**
```
Given I am on any page of the site
When I type "wireless headphones" into the search bar and press Enter
Then I see a results page showing products matching that keyword
  And results are sorted by relevance by default
  And the total number of results is displayed
```

**Scenario 2: Filter by price range**
```
Given I have search results displayed on screen
When I set a price range of $20–$100 and apply the filter
Then only products priced between $20 and $100 are shown
  And the active filter is visually indicated
  And I can remove the filter to restore full results
```

**Scenario 3: No results found**
```
Given I search for a term that matches no products
When the search completes
Then I see a "No results found" message
  And I am shown suggested alternative search terms or popular categories
```

---

## Example 3: Order Notifications

**Title:** Receive real-time order status notifications

As a customer who has placed an order,
I want to receive notifications when my order status changes,
So that I can track my delivery without manually checking the site.

**Priority:** Should Have
**Story Points:** 3
**Sprint:** Sprint 4

### Acceptance Criteria

**Scenario 1: Notification sent when order ships**
```
Given my order status changes to "Shipped"
When the fulfillment system updates the order record
Then I receive an email notification within 5 minutes
  And the email includes the tracking number and carrier name
  And a tracking link is provided
```

**Scenario 2: User opts out of notifications**
```
Given I have opted out of marketing and order emails in my preferences
When my order status changes
Then I do not receive any email notifications
  And I can still view order status by logging in to my account
```

---

## INVEST Criteria Checklist

Use this checklist to validate a user story before adding it to the sprint backlog.

| Criterion   | Question to Ask                                        | Check |
|-------------|--------------------------------------------------------|-------|
| Independent | Can this story be developed without depending on another unfinished story? | [ ] |
| Negotiable  | Is there room to discuss implementation details with the team? | [ ] |
| Valuable    | Does this story deliver clear value to the user or business? | [ ] |
| Estimable   | Does the team have enough information to estimate it?  | [ ] |
| Small       | Can it be completed within a single sprint?            | [ ] |
| Testable    | Are the acceptance criteria specific and verifiable?   | [ ] |

---

## Definition of Done (DoD)

A story is considered complete when all of the following are true:

- [ ] All acceptance criteria scenarios pass
- [ ] Code is reviewed and merged to main branch
- [ ] Unit and integration tests written and passing
- [ ] No new linting or static analysis errors introduced
- [ ] Feature tested in staging environment
- [ ] Documentation or release notes updated if applicable
- [ ] Product Owner has accepted the story
