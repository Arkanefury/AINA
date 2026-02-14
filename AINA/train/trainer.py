file_path = "/"

if not os.path.exists(file_path):
    print(f"ERROR: not found {file_path}")
else:
    dataset = MyChatDataset(file_path, tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_dataloader) * epochs)

    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_dataloader, leave=True)
        total_loss = 0

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(train_dataloader)}")

        model.save_pretrained("/")
        tokenizer.save_pretrained("/")