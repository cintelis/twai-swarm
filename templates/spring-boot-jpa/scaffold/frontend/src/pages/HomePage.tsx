import { useNavigate } from 'react-router-dom'
import { useDispatch } from 'react-redux'
import { useHelloQuery, useMeQuery } from '../api/api'
import { clearToken } from '../api/authSlice'

export default function HomePage() {
  const navigate = useNavigate()
  const dispatch = useDispatch()
  const { data: hello, isLoading: helloLoading } = useHelloQuery()
  const { data: me, isLoading: meLoading, error: meError } = useMeQuery()

  const handleLogout = () => {
    dispatch(clearToken())
    navigate('/login')
  }

  return (
    <div className="card">
      <h1>Home</h1>
      <p className="subtitle">Replace this page with your app's main UI.</p>

      <section data-testid="hello-section">
        <h2>Public endpoint</h2>
        {helloLoading && <p>Loading…</p>}
        {hello && (
          <pre data-testid="hello-payload">{JSON.stringify(hello, null, 2)}</pre>
        )}
      </section>

      <section data-testid="me-section">
        <h2>Protected endpoint</h2>
        {meLoading && <p>Loading…</p>}
        {meError && (
          <p data-testid="me-error" className="error-message">
            Could not load /api/hello/me — are you signed in?
          </p>
        )}
        {me && (
          <p data-testid="me-username">Signed in as <strong>{me.username}</strong></p>
        )}
      </section>

      <button
        data-testid="logout-button"
        className="btn btn-primary"
        onClick={handleLogout}
      >
        Sign out
      </button>
    </div>
  )
}
