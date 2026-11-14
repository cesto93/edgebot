package cmd

import "testing"

// TestSupersededRequestCannotMutateUI guards the request-generation logic: a
// slow goroutine from a cancelled request must not append its error or clear
// the loading state of the request that replaced it.
func TestSupersededRequestCannotMutateUI(t *testing.T) {
	m := &ShellModel{}

	first := m.beginRequest()
	second := m.beginRequest()
	if first == second {
		t.Fatal("beginRequest returned the same id twice")
	}

	m.failWithError(first, "stale error")
	if len(m.messages) != 0 {
		t.Fatalf("stale request appended %d messages, want 0", len(m.messages))
	}
	if !m.loading {
		t.Fatal("stale request cleared the loading flag of the active request")
	}

	m.finishRequest(first)
	if !m.loading {
		t.Fatal("stale cancellation cleared the loading flag of the active request")
	}

	m.finishRequest(second)
	if m.loading {
		t.Fatal("active request did not clear the loading flag")
	}

	m.failWithError(second, "real error")
	if len(m.messages) != 1 || m.messages[0].role != "error" {
		t.Fatalf("active request did not append its error: %+v", m.messages)
	}
}
