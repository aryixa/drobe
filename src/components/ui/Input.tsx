import React from 'react'
import { clsx } from 'clsx'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  icon?: React.ReactNode
}

export function Input({
  label,
  error,
  icon,
  className,
  ...props
}: InputProps) {
  const baseClasses = 'block w-full px-3 py-2 border rounded-md shadow-sm placeholder-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 sm:text-sm transition-colors duration-200'
  
  const stateClasses = error
    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
    : 'border-primary-300'

  const classes = clsx(baseClasses, stateClasses, className)

  return (
    <div className="w-full">
      {label && (
        <label className="block text-sm font-medium text-primary-700 mb-1">
          {label}
        </label>
      )}
      <div className="relative">
        {icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <div className="text-primary-400">{icon}</div>
          </div>
        )}
        <input
          className={clsx(
            classes,
            icon && 'pl-10'
          )}
          {...props}
        />
      </div>
      {error && (
        <p className="mt-1 text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  )
}
