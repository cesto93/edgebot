package cmd

import (
	"fmt"
	"strings"

	"edgebot/llm"
)

// lastAssistantText returns the trimmed string content of the final message,
// or an error when the LLM returned no messages or an empty/missing response.
func lastAssistantText(resultMessages []llm.Message) (string, error) {
	if len(resultMessages) == 0 {
		return "", fmt.Errorf("no response from LLM")
	}

	lastMsg := resultMessages[len(resultMessages)-1]
	content, ok := lastMsg.Content.(string)
	if !ok || strings.TrimSpace(content) == "" {
		return "", fmt.Errorf("empty response from LLM")
	}

	return strings.TrimSpace(content), nil
}
